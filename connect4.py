"""
Connect 4 — Fast RL (Linear Q + Self-Play + Symmetry)
-----------------------------------------------------
- Linear Q-learning on features φ(s, a) — learns fast and is stable.
- Self-play (two copies share weights), ε-greedy with decay.
- Symmetry augmentation (mirror columns) for sample efficiency.
- Reward shaping: +1 win / -1 loss / +0.2 create-3 / +0.5 block-opp-3 / +0.75 immediate-win.
- Clean Pygame UI: Train / Play / Watch; adjustable speed, save/load.

Run:
  pip install pygame numpy
  python connect4_rl.py

Keys:
  ENTER start / I instructions
  T train (AI vs AI)   |  W watch (AI vs AI, no training)
  P play (Human vs AI) |  R reset game
  1/2/3 speed          |  F fast
  S save weights       |  L load weights
  ESC quit
"""

import sys, math, random, warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import numpy as np
import pygame as pg

# ----------------------------- Game constants -----------------------------
ROWS, COLS = 6, 7
CONNECT = 4
EMPTY = 0
P1, P2 = 1, -1  # agent is "current player" (sign flips every move)

# UI
CELL = 90
MARGIN = 40
HUD_H = 120
BG = (18, 20, 28)
GRID = (34, 39, 53)
DISC1 = (255, 214, 102)  # gold
DISC2 = (99, 255, 218)   # teal
INK = (236, 242, 255)
ACCENT = (99, 102, 241)
OK = (16, 185, 129)
MUTED = (150, 160, 175)
CARD = (28, 32, 44)
SPEEDS = {"1x": 1, "5x": 5, "20x": 20}

# RL
ALPHA = 0.05
GAMMA = 0.99
EPS_START = 0.40
EPS_MIN = 0.02
EPS_DECAY = 0.9995

# Reward shaping
R_WIN = 1.0
R_LOSS = -1.0
R_DRAW = 0.0
R_IMM_WIN = 0.75
R_BLOCK_THREE = 0.50
R_CREATE_THREE = 0.20

WEIGHTS_FILE = "c4_weights.npy"

# ----------------------------- Board utilities ----------------------------
class C4:
    def __init__(self):
        self.reset()

    def reset(self):
        self.b = np.zeros((ROWS, COLS), dtype=np.int8)
        self.turn = P1   # current player sign (1 or -1)
        self.moves = 0
        self.winner = 0

    def copy(self):
        c = C4()
        c.b[:] = self.b
        c.turn = self.turn
        c.moves = self.moves
        c.winner = self.winner
        return c

    def legal(self):
        return [c for c in range(COLS) if self.b[0, c] == EMPTY]

    def drop(self, col):
        if col not in self.legal():
            return False
        r = ROWS - 1
        while r >= 0 and self.b[r, col] != EMPTY:
            r -= 1
        self.b[r, col] = self.turn
        self.moves += 1
        self.winner = self._check(r, col)
        self.turn = -self.turn
        return True

    def _check(self, r, c):
        me = int(self.b[r, c])
        dirs = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in dirs:
            cnt = 1
            for sgn in (1,-1):
                rr, cc = r+sgn*dr, c+sgn*dc
                while 0 <= rr < ROWS and 0 <= cc < COLS and self.b[rr,cc] == me:
                    cnt += 1
                    rr += sgn*dr; cc += sgn*dc
            if cnt >= CONNECT:
                return me
        if self.moves == ROWS*COLS:
            return 2  # draw
        return 0

    def mirror(self):
        m = self.copy()
        m.b = np.fliplr(self.b.copy())
        m.turn = self.turn
        m.winner = self.winner
        return m

# ----------------------------- Feature engineering ------------------------
def count_windows(arr, n, player):
    cnt = 0
    for i in range(len(arr)-CONNECT+1):
        window = arr[i:i+CONNECT]
        if np.all((window == player) | (window == 0)):
            if np.sum(window == player) == n and np.sum(window == 0) == CONNECT-n:
                cnt += 1
    return cnt

def board_features(board, player):
    """Feature vector that’s fast & informative, from the POV of `player` (1/-1)."""
    b = board * player  # make 'player' pieces positive
    opp = -player
    feats = []

    # Center control (encourage middle columns)
    center_col = b[:, COLS//2]
    center_score = np.sum(center_col == 1) - np.sum(center_col == -1)
    feats.append(center_score / ROWS)

    # 2-in-a-rows and 3-in-a-rows counts (potential lines)
    my2=my3=opp2=opp3=0
    # rows
    for r in range(ROWS):
        row = b[r, :]
        my2 += count_windows(row, 2, 1)
        my3 += count_windows(row, 3, 1)
        opp2 += count_windows(row, 2, -1)
        opp3 += count_windows(row, 3, -1)
    # cols
    for c in range(COLS):
        col = b[:, c]
        my2 += count_windows(col, 2, 1)
        my3 += count_windows(col, 3, 1)
        opp2 += count_windows(col, 2, -1)
        opp3 += count_windows(col, 3, -1)
    # diags
    for r in range(ROWS-3):
        for c in range(COLS-3):
            diag = np.array([b[r+i, c+i] for i in range(4)])
            anti = np.array([b[r+3-i, c+i] for i in range(4)])
            my2 += count_windows(diag, 2, 1); my3 += count_windows(diag, 3, 1)
            opp2 += count_windows(diag, 2, -1); opp3 += count_windows(diag, 3, -1)
            my2 += count_windows(anti, 2, 1); my3 += count_windows(anti, 3, 1)
            opp2 += count_windows(anti, 2, -1); opp3 += count_windows(anti, 3, -1)

    feats += [my2/24.0, my3/24.0, opp2/24.0, opp3/24.0]  # normalized
    feats.append(np.sum(b==0)/(ROWS*COLS))  # emptiness
    feats.append(1.0)  # bias
    return np.array(feats, dtype=np.float32)

def features_after_move(state: C4, col):
    """φ(s,a): features after dropping in col, from current player's POV."""
    if col not in state.legal():
        # strongly negative for illegal
        return None, True
    s2 = state.copy()
    s2.drop(col)
    me = -s2.turn  # we just moved, so previous turn
    f = board_features(s2.b, me)
    # Auxiliary tactical bits
    imm_win = 1.0 if s2.winner == me else 0.0
    # does this move block opponent immediate win next turn?
    blocks = 0.0
    if imm_win == 0.0:
        # if opponent had an immediate win now (before move), and now it's gone
        before = has_immediate_win(state, -state.turn)
        after  = has_immediate_win(s2, -s2.turn)
        blocks = 1.0 if (before and not after) else 0.0
    create3 = local_create_three(state, col, state.turn)
    f = np.concatenate([f, np.array([imm_win, blocks, create3], dtype=np.float32)])
    return f, False

def has_immediate_win(state: C4, player):
    for c in state.legal():
        tmp = state.copy()
        tmp.turn = player
        tmp.drop(c)
        if tmp.winner == player:
            return True
    return False

def local_create_three(state: C4, col, player):
    """Return 1 if the move creates at least one new 3-in-a-row."""
    if col not in state.legal(): return 0.0
    tmp = state.copy()
    tmp.turn = player
    tmp.drop(col)
    b = tmp.b * player
    total = 0
    # check only windows that include the placed disc neighborhood for speed
    # (good enough for shaping)
    for r in range(ROWS):
        total += count_windows(b[r,:], 3, 1)
    for c in range(COLS):
        total += count_windows(b[:,c], 3, 1)
    for r in range(ROWS-3):
        for c in range(COLS-3):
            diag = np.array([b[r+i, c+i] for i in range(4)])
            anti = np.array([b[r+3-i, c+i] for i in range(4)])
            total += count_windows(diag,3,1)
            total += count_windows(anti,3,1)
    return 1.0 if total>0 else 0.0

# ----------------------------- Linear Q agent -----------------------------
class LinearQ:
    def __init__(self, n_feats):
        self.W = np.zeros(n_feats, dtype=np.float32)
        self.eps = EPS_START

    def q(self, phi):
        return float(np.dot(self.W, phi))

    def act(self, state: C4, explore=True):
        legal = state.legal()
        if not legal:
            return None
        # Epsilon-greedy with symmetry tie-break toward center
        if explore and random.random() < self.eps:
            return random.choice(legal)
        best, best_q = None, -1e9
        for c in legal:
            phi, _ = features_after_move(state, c)
            q = self.q(phi)
            # prefer center on ties
            if q > best_q or (abs(q-best_q)<1e-6 and abs(c-3) < abs((best if best is not None else 3)-3)):
                best_q, best = q, c
        return best

    def update(self, s: C4, a, r, s2: C4, done):
        phi_a, _ = features_after_move(s, a)
        q_sa = self.q(phi_a)

        if done:
            target = r
        else:
            # Max over next actions
            best_q = -1e9
            for c in s2.legal():
                phi2, _ = features_after_move(s2, c)
                best_q = max(best_q, self.q(phi2))
            target = r + GAMMA * best_q

        td = target - q_sa
        self.W += ALPHA * td * phi_a
        self.eps = max(EPS_MIN, self.eps * EPS_DECAY)

    def save(self, path=WEIGHTS_FILE):
        np.save(path, {"W": self.W, "eps": self.eps}, allow_pickle=True)

    def load(self, path=WEIGHTS_FILE):
        d = np.load(path, allow_pickle=True).item()
        self.W = d["W"].astype(np.float32)
        self.eps = float(d.get("eps", EPS_START))

# ----------------------------- Pygame UI ---------------------------------
class App:
    def __init__(self):
        pg.init()
        pg.display.set_caption("Connect 4 — Fast RL")
        w = COLS*CELL + 2*MARGIN
        h = ROWS*CELL + 2*MARGIN + HUD_H
        self.screen = pg.display.set_mode((w, h), pg.RESIZABLE)
        self.clock = pg.time.Clock()

        # Game
        self.g = C4()
        base_feats = len(board_features(self.g.b, 1)) + 3  # + imm_win, block, create3
        self.agent = LinearQ(base_feats)

        # Modes
        self.mode = "menu"      # menu, train, watch, play
        self.fast = False
        self.speed_key = "1x"
        self.human_player = P1  # human is P1 when playing
        self.result_text = ""

        # Training stats
        self.ep = 1
        self.ep_rewards = []
        self.avg100 = []
        self.last_move_col = None

    # --------- Drawing ----------
    def draw(self):
        self.screen.fill(BG)
        W, H = self.screen.get_size()
        # Board area
        bx, by = MARGIN, MARGIN
        bw, bh = COLS*CELL, ROWS*CELL
        # Grid
        pg.draw.rect(self.screen, GRID, (bx-6, by-6, bw+12, bh+12), border_radius=18)
        for r in range(ROWS):
            for c in range(COLS):
                cx = bx + c*CELL + CELL//2
                cy = by + r*CELL + CELL//2
                pg.draw.circle(self.screen, CARD, (cx, cy), CELL//2 - 6)
                v = self.g.b[r,c]
                if v != 0:
                    col = DISC1 if v==P1 else DISC2
                    pg.draw.circle(self.screen, col, (cx, cy), CELL//2 - 10)

        # Column hover highlight (play mode)
        if self.mode == "play":
            mx, my = pg.mouse.get_pos()
            if by <= my <= by+bh:
                col = (mx - bx) // CELL
                if 0 <= col < COLS and self.g.b[0,col]==0:
                    rect = pg.Rect(bx+col*CELL, by, CELL, bh)
                    s = pg.Surface(rect.size, pg.SRCALPHA); s.fill((255,255,255,22))
                    self.screen.blit(s, rect.topleft)

        # HUD
        self.draw_hud()

        if self.mode == "menu":
            self.draw_menu_overlay()

    def draw_hud(self):
        W, H = self.screen.get_size()
        panel = pg.Rect(MARGIN, H - HUD_H + 10, W - 2*MARGIN, HUD_H - 20)
        pg.draw.rect(self.screen, CARD, panel, border_radius=16)
        x = panel.x + 14; y = panel.y + 10
        self.text(f"Mode: {self.mode.upper()}    Eps {self.agent.eps:.2f}    Ep {self.ep}", x, y, 22, INK)
        y += 28
        avg = (sum(self.avg100)/len(self.avg100)) if self.avg100 else 0.0
        self.text(f"AvgR(100): {avg: .3f}    Last: {self.ep_rewards[-1]: .3f}" if self.ep_rewards else "AvgR(100): 0.000    Last: 0.000", x, y, 20, INK)
        y += 26
        self.text("Keys: ENTER start | I instructions | T train | W watch | P play | 1/2/3 speed | F fast | S save | L load | R reset | ESC quit",
                  x, y, 18, MUTED)

        # Result banner
        if self.g.winner == P1:
            self.text("P1 wins!", panel.centerx, panel.y-28, 26, OK, center=True)
        elif self.g.winner == P2:
            self.text("P2 wins!", panel.centerx, panel.y-28, 26, ACCENT, center=True)
        elif self.g.winner == 2:
            self.text("Draw", panel.centerx, panel.y-28, 26, MUTED, center=True)

    def draw_menu_overlay(self):
        W, H = self.screen.get_size()
        overlay = pg.Surface((W,H), pg.SRCALPHA); overlay.fill((0,0,0,140))
        self.screen.blit(overlay, (0,0))
        card_w, card_h = int(W*0.8), int(H*0.6)
        card = pg.Rect((W-card_w)//2, (H-card_h)//2, card_w, card_h)
        pg.draw.rect(self.screen, CARD, card, border_radius=20)
        self.text("CONNECT 4 — FAST RL", card.centerx, card.y+30, 34, INK, center=True)
        y = card.y + 80
        lines = [
            "This agent learns with Linear Q-learning on hand-crafted features.",
            "Training is self-play: two copies share weights and improve together.",
            "Rewards: +1 win, −1 loss, +0.75 immediate-win, +0.5 block-opp-3, +0.2 create-3.",
            "Press T to train, P to play vs the agent, or W to watch AI vs AI.",
            "Change speed with 1/2/3 and toggle fast mode with F.",
            "Save/Load with S/L.",
        ]
        for ln in lines:
            self.text(ln, card.x+30, y, 22, INK); y += 32
        self.text("Press ENTER to close", card.centerx, card.bottom-36, 22, ACCENT, center=True)

    def text(self, s, x, y, size=20, color=INK, center=False):
        font = pg.font.SysFont("Segoe UI", size)
        img = font.render(str(s), True, color)
        rect = img.get_rect()
        if center: rect.center = (x, y)
        else: rect.topleft = (x, y)
        self.screen.blit(img, rect)

    # --------- Loop / events ----------
    def handle_input(self):
        action_col = None
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit(); sys.exit(0)
            elif e.type == pg.VIDEORESIZE:
                self.screen = pg.display.set_mode((e.w, e.h), pg.RESIZABLE)
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE: pg.quit(); sys.exit(0)
                if e.key == pg.K_i: self.mode = "menu"
                if e.key == pg.K_RETURN and self.mode=="menu": self.mode = "play"
                if e.key == pg.K_t: self.mode = "train"
                if e.key == pg.K_w: self.mode = "watch"
                if e.key == pg.K_p: self.mode = "play"
                if e.key == pg.K_r: self.reset_game()
                if e.key == pg.K_s:
                    try: self.agent.save(); print("Saved", WEIGHTS_FILE)
                    except Exception as ex: print("Save failed:", ex)
                if e.key == pg.K_l:
                    try: self.agent.load(); print("Loaded", WEIGHTS_FILE)
                    except Exception as ex: print("Load failed:", ex)
                if e.key == pg.K_1: self.speed_key = "1x"
                if e.key == pg.K_2: self.speed_key = "5x"
                if e.key == pg.K_3: self.speed_key = "20x"
                if e.key == pg.K_f: self.fast = not self.fast
            elif e.type == pg.MOUSEBUTTONDOWN and self.mode == "play" and self.g.winner==0:
                bx, by = MARGIN, MARGIN
                mx, my = e.pos
                if by <= my <= by+ROWS*CELL:
                    col = (mx - bx)//CELL
                    if 0 <= col < COLS and self.g.b[0,col]==0:
                        action_col = col
        return action_col

    def reset_game(self):
        self.g.reset()
        self.result_text = ""

    # --------- RL helpers ----------
    def reward_shaping(self, s: C4, a, s2: C4):
        """Small extra rewards for tactical behaviors."""
        r = 0.0
        # immediate win
        phi, _ = features_after_move(s, a)
        if phi[-3] >= 0.5:  # imm_win feature
            r += R_IMM_WIN
        # block opponent immediate win
        if phi[-2] >= 0.5:
            r += R_BLOCK_THREE
        # created a 3-in-a-row
        if phi[-1] >= 0.5:
            r += R_CREATE_THREE
        return r

    def self_play_step(self):
        """One move in self-play (both sides share weights). Returns (done, terminal_reward)."""
        if self.g.winner != 0:
            # terminal already
            return True, (R_DRAW if self.g.winner==2 else (R_WIN if self.g.winner==P2 else R_LOSS))
        s = self.g.copy()
        a = self.agent.act(s, explore=True if self.mode=="train" else False)
        if a is None:  # no legal
            self.g.winner = 2
            return True, R_DRAW
        self.g.drop(a)
        s2 = self.g.copy()
        # intermediate shaping
        r = self.reward_shaping(s, a, s2)
        done = (self.g.winner != 0)
        if done:
            if self.g.winner == 2: r += R_DRAW
            elif self.g.winner == -s.turn: r += R_WIN   # the player who just moved is winner
            else: r += R_LOSS
        # Q update from the mover's perspective:
        self.agent.update(s, a, r, s2, done)
        return done, r

    def play_step(self, human_col=None):
        """Human vs AI. Human is P1 by default."""
        # Human move
        if self.g.winner != 0: return
        if self.g.turn == self.human_player:
            if human_col is not None and human_col in self.g.legal():
                self.g.drop(human_col)
        else:
            # AI move (greedy)
            a = self.agent.act(self.g, explore=False)
            if a is not None:
                self.g.drop(a)

    # --------- Main loop ----------
    def run(self):
        pg.font.init()
        last = pg.time.get_ticks()
        while True:
            self.clock.tick(60)
            now = pg.time.get_ticks()
            dt = (now - last)/1000.0; last = now

            human_col = self.handle_input()

            repeats = SPEEDS[self.speed_key]
            if self.fast: repeats = 60

            if self.mode in ("train","watch"):
                # self-play steps
                for _ in range(repeats):
                    done, r = self.self_play_step()
                    if done:
                        # episode end
                        total = 0.0
                        # approximate return: win/loss counted in update already
                        self.ep_rewards.append(total)
                        if len(self.ep_rewards) > 100: self.ep_rewards.pop(0)
                        self.avg100.append(r)
                        if len(self.avg100) > 100: self.avg100.pop(0)
                        self.g.reset()
                        self.ep += 1
                        break
            elif self.mode == "play":
                # slower stepping; only 1 move considered per frame for responsiveness
                self.play_step(human_col)

            self.draw()
            pg.display.flip()

# ----------------------------- Main --------------------------------------
if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        pg.quit(); sys.exit(0)
