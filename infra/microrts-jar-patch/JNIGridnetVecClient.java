/*
 * Patched JNIGridnetVecClient for gym-microrts 0.3.2's vendored microrts.jar.
 *
 * Upstream bug this fixes: JNIGridnetClient reuses ONE Response object per env;
 * its reset() overwrites Response.observation with the new episode's first
 * frame. gameStep() here auto-resets terminated envs BEFORE control returns to
 * Python, so the terminal (pre-reset) board state is never observable — the
 * done-step observation is byte-identical to the next episode's first frame.
 *
 * Patch 1 (terminal obs): capture a deep copy of each lane's observation into
 * the public `terminalObservation` field right before the auto-reset fires
 * (both bot and self-play lanes). Lanes are only refreshed on termination;
 * Python must read terminalObservation[i] only when done[i] is true.
 *
 * Patch 2 (opponent action): JNIGridnetClient keeps the scripted bot's
 * PlayerAction in its package-visible `pa2` field but never surfaces it over
 * JNI. After every bot-lane gameStep we encode `pa2` into the same per-cell
 * gridnet layout Python actions use — `(H*W, 7)` with component order
 * [action_type, move_dir, harvest_dir, return_dir, produce_dir, produce_type,
 * attack_offset]; cells with no acting unit stay all-zero (NOOP) — the exact
 * inverse of UnitAction.fromActionArray. Exposed as the public
 * `opponentAction` field, same tick as the learner's submitted action (pa1 and
 * pa2 are issued in the same engine cycle). Self-play lanes stay zero: both
 * actions are already in Python's hands.
 *
 * Patch 3 (per-lane seats): reset(int[])/gameStep(..., int[]) already take
 * per-lane player ids but gym_microrts hardcodes zeros, so the Python policy
 * could only ever be player 0. The patched client keeps a `playerIds` array
 * (settable via setPlayerIds BEFORE reset) and uses it for bot lanes in
 * reset/gameStep/getMasks, ignoring the zeros the wrapper passes. Setting
 * playerIds[i]=1 puts the scripted bot in the player-0 seat of lane i, with
 * obs/masks/rewards still from the Python player's perspective — role-swapped
 * data collection without touching gym_microrts. Self-play lanes are unaffected.
 *
 * Everything else is behavior-identical to the decompiled original (CFR 0.152).
 *
 * Build/apply: see apply_patch.sh next to this file.
 */
package tests;

import ai.PassiveAI;
import ai.core.AI;
import ai.rewardfunction.RewardFunctionInterface;
import rts.PlayerAction;
import rts.GameState;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import tests.JNIGridnetClient;
import tests.JNIGridnetClientSelfPlay;
import util.Pair;

public class JNIGridnetVecClient {
    public JNIGridnetClient[] clients;
    public JNIGridnetClientSelfPlay[] selfPlayClients;
    public int maxSteps;
    public int[] envSteps;
    public RewardFunctionInterface[] rfs;
    public UnitTypeTable utt;
    int[][][][] masks;
    int[][][][] observation;
    double[][] reward;
    boolean[][] done;
    JNIGridnetClient.Response[] rs;
    Responses responses;
    double[] terminalReward1;
    boolean[] terminalRone1;
    double[] terminalReward2;
    boolean[] terminalRone2;
    // Pre-reset terminal arrival frame per lane, refreshed only when that lane
    // terminates (done[0] or maxSteps). Zero-initialized; valid iff done[i].
    public int[][][][] terminalObservation;
    // Scripted opponent's action per lane, gridnet-encoded (H*W, 7); refreshed
    // every gameStep for bot lanes, all-zero for self-play lanes and after reset.
    public int[][][] opponentAction;
    // Markov-complete structured state for world-model v2. Cell rows are
    // row-major (H*W, 16); globals are (8). Both are canonicalized around the
    // lane's Python-controlled player. See docs/micro-rts/
    // WORLD_MODEL_V2_REPRESENTATION.md for the field contract.
    public int[][][] fullState;
    public int[][] fullGlobals;
    // Pre-reset terminal structured state, valid only when done[i] is true.
    public int[][][] terminalFullState;
    public int[][] terminalFullGlobals;
    // Counterfactual one-step arrival populated by computeCounterfactual().
    public int[][][] counterfactualFullState;
    public int[][] counterfactualFullGlobals;
    private GameState[] counterfactualSource;
    // Per-lane player id of the Python-controlled player (bot lanes only; 0 or 1).
    public int[] playerIds;
    int gridWidth;
    int gridCells;
    int attackDiameter;

    public JNIGridnetVecClient(int n, int n2, int n3, RewardFunctionInterface[] rewardFunctionInterfaceArray, String string, String string2, AI[] aIArray, UnitTypeTable unitTypeTable) throws Exception {
        int n4;
        this.maxSteps = n3;
        this.utt = unitTypeTable;
        this.rfs = rewardFunctionInterfaceArray;
        this.envSteps = new int[n + n2];
        this.selfPlayClients = new JNIGridnetClientSelfPlay[n / 2];
        for (n4 = 0; n4 < this.selfPlayClients.length; ++n4) {
            this.selfPlayClients[n4] = new JNIGridnetClientSelfPlay(rewardFunctionInterfaceArray, string, string2, unitTypeTable);
        }
        this.clients = new JNIGridnetClient[n2];
        for (n4 = 0; n4 < this.clients.length; ++n4) {
            this.clients[n4] = new JNIGridnetClient(rewardFunctionInterfaceArray, string, string2, aIArray[n4], unitTypeTable);
        }
        JNIGridnetClient.Response response = new JNIGridnetClient(rewardFunctionInterfaceArray, string, string2, new PassiveAI(unitTypeTable), unitTypeTable).reset(0);
        int n5 = n + n2;
        int n6 = response.observation.length;
        int n7 = response.observation[0].length;
        int n8 = response.observation[0][0].length;
        this.masks = new int[n5][][][];
        this.observation = new int[n5][n6][n7][n8];
        this.reward = new double[n5][this.rfs.length];
        this.done = new boolean[n5][this.rfs.length];
        this.terminalReward1 = new double[this.rfs.length];
        this.terminalRone1 = new boolean[this.rfs.length];
        this.terminalReward2 = new double[this.rfs.length];
        this.terminalRone2 = new boolean[this.rfs.length];
        this.terminalObservation = new int[n5][n6][n7][n8];
        // observation is (planes, H, W): n7 = H, n8 = W.
        this.gridWidth = n8;
        this.gridCells = n7 * n8;
        this.attackDiameter = unitTypeTable.getMaxAttackRange() * 2 + 1;
        this.opponentAction = new int[n5][this.gridCells][7];
        this.fullState = new int[n5][this.gridCells][16];
        this.fullGlobals = new int[n5][8];
        this.terminalFullState = new int[n5][this.gridCells][16];
        this.terminalFullGlobals = new int[n5][8];
        this.counterfactualFullState = new int[n5][this.gridCells][16];
        this.counterfactualFullGlobals = new int[n5][8];
        this.counterfactualSource = new GameState[n5];
        this.playerIds = new int[n5];
        this.responses = new Responses(null, null, null);
        this.rs = new JNIGridnetClient.Response[n5];
    }

    /** Per-lane seat of the Python-controlled player (bot lanes; 0 or 1 each).
     * Call BEFORE reset; self-play lanes ignore their entries. */
    public void setPlayerIds(int[] ids) {
        for (int i = 0; i < this.playerIds.length && i < ids.length; ++i) {
            this.playerIds[i] = ids[i];
        }
    }

    /** Inverse of UnitAction.fromActionArray: PlayerAction -> (H*W, 7) gridnet. */
    private int[][] encodePlayerAction(PlayerAction pa) {
        int[][] out = new int[this.gridCells][7];
        if (pa == null) {
            return out;
        }
        int r = this.attackDiameter;
        int half = r / 2;
        for (Pair<Unit, UnitAction> p : pa.getActions()) {
            Unit u = p.m_a;
            UnitAction ua = p.m_b;
            if (u == null || ua == null) continue;
            int cell = u.getY() * this.gridWidth + u.getX();
            if (cell < 0 || cell >= this.gridCells) continue;
            int[] a = out[cell];
            int type = ua.getType();
            if (type < 0 || type > UnitAction.TYPE_ATTACK_LOCATION) continue;
            a[0] = type;
            // TYPE_NONE (0) otherwise serializes identically to an absent
            // action. Component 6 is inactive for NONE, and 255 is outside the
            // normal 7x7 attack-offset range, so it is an unambiguous archival
            // issued-action marker consumed only by the Python v2 schema.
            if (type == UnitAction.TYPE_NONE) a[6] = 255;
            switch (type) {
                case UnitAction.TYPE_MOVE: {
                    a[1] = Math.max(ua.getDirection(), 0);
                    break;
                }
                case UnitAction.TYPE_HARVEST: {
                    a[2] = Math.max(ua.getDirection(), 0);
                    break;
                }
                case UnitAction.TYPE_RETURN: {
                    a[3] = Math.max(ua.getDirection(), 0);
                    break;
                }
                case UnitAction.TYPE_PRODUCE: {
                    a[4] = Math.max(ua.getDirection(), 0);
                    a[5] = ua.getUnitType() != null ? ua.getUnitType().ID : 0;
                    break;
                }
                case UnitAction.TYPE_ATTACK_LOCATION: {
                    int dx = ua.getLocationX() - u.getX();
                    int dy = ua.getLocationY() - u.getY();
                    int off = (dy + half) * r + (dx + half);
                    if (off >= 0 && off < r * r) {
                        a[6] = off;
                    } else {
                        a[0] = UnitAction.TYPE_NONE;   // out of gridnet range
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
        return out;
    }

    private static int[][][] copyObs(int[][][] src) {
        int[][][] dst = new int[src.length][][];
        for (int i = 0; i < src.length; ++i) {
            dst[i] = new int[src[i].length][];
            for (int j = 0; j < src[i].length; ++j) {
                dst[i][j] = src[i][j].clone();
            }
        }
        return dst;
    }

    private static int[][] copy2d(int[][] src) {
        int[][] dst = new int[src.length][];
        for (int i = 0; i < src.length; ++i) dst[i] = src[i].clone();
        return dst;
    }

    /** Encode a lossless-enough Markov snapshot around ``perspective``.
     * Empty/sentinel values intentionally use -1; HDF5 v4 stores int32. */
    private int[][] encodeFullState(GameState gs, int perspective) {
        int[][] out = new int[this.gridCells][16];
        int h = gs.getPhysicalGameState().getHeight();
        int w = gs.getPhysicalGameState().getWidth();
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int cell = y * w + x;
                int[] row = out[cell];
                row[0] = gs.getPhysicalGameState().getTerrain(x, y);
                row[2] = -1;
                row[3] = -1;
                row[4] = -1;
                row[8] = -1;
                row[9] = -1;
                row[10] = -1;
                row[11] = -1;
                row[12] = -1;
                row[13] = -1;
            }
        }
        for (Unit u : gs.getUnits()) {
            int cell = u.getY() * w + u.getX();
            if (cell < 0 || cell >= out.length) continue;
            int[] row = out[cell];
            row[1] = 1;
            long uid = u.getID();
            row[2] = uid > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int)uid;
            row[3] = u.getPlayer() < 0 ? 0 : (u.getPlayer() == perspective ? 1 : 2);
            row[4] = u.getType().ID;
            row[5] = u.getHitPoints();
            row[6] = u.getResources();
            UnitActionAssignment aa = gs.getActionAssignment(u);
            if (aa == null || aa.action == null) continue;
            UnitAction ua = aa.action;
            int type = ua.getType();
            row[7] = 1;
            row[8] = type;
            if (type == UnitAction.TYPE_MOVE || type == UnitAction.TYPE_HARVEST ||
                type == UnitAction.TYPE_RETURN || type == UnitAction.TYPE_PRODUCE) {
                int dir = ua.getDirection();
                row[9] = dir;
                if (dir >= 0 && dir < UnitAction.DIRECTION_OFFSET_X.length) {
                    row[10] = u.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
                    row[11] = u.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
                }
            } else if (type == UnitAction.TYPE_ATTACK_LOCATION) {
                row[10] = ua.getLocationX();
                row[11] = ua.getLocationY();
            }
            row[12] = ua.getUnitType() == null ? -1 : ua.getUnitType().ID;
            row[13] = aa.time;
            int eta = ua.ETA(u);
            row[14] = eta;
            row[15] = Math.max(0, aa.time + eta - gs.getTime());
        }
        return out;
    }

    private int[] encodeFullGlobals(GameState gs, int perspective) {
        int[] out = new int[8];
        int other = perspective == 0 ? 1 : 0;
        ResourceUsage ru = gs.getResourceUsage();
        out[0] = gs.getTime();
        out[1] = gs.getPlayer(perspective).getResources();
        out[2] = gs.getPlayer(other).getResources();
        out[3] = ru.getResourcesUsed(perspective);
        out[4] = ru.getResourcesUsed(other);
        out[5] = ru.getPositionsUsed().size();
        int winner = gs.winner();
        out[6] = winner < 0 ? -1 : (winner == perspective ? 1 : 2);
        out[7] = gs.gameover() ? 1 : 0;
        return out;
    }

    private void refreshFullState(int lane, GameState gs, int perspective) {
        this.fullState[lane] = encodeFullState(gs, perspective);
        this.fullGlobals[lane] = encodeFullGlobals(gs, perspective);
    }

    /** Clone every lane, issue the supplied alternative joint action, cycle it
     * once, and export the arrival without mutating the live environment. The
     * two action arrays are perspective-relative: self first, opponent second. */
    public void computeCounterfactual(int[][][] selfActions,
                                      int[][][] opponentActions,
                                      boolean[] valid) throws Exception {
        int selfLanes = this.selfPlayClients.length * 2;
        for (int lane = 0; lane < this.rs.length; ++lane) {
            if (lane >= valid.length || !valid[lane]) {
                this.counterfactualFullState[lane] = new int[this.gridCells][16];
                this.counterfactualFullGlobals[lane] = new int[8];
                continue;
            }
            int perspective;
            GameState source;
            ai.jni.JNIInterface iface;
            if (lane < selfLanes) {
                int pair = lane / 2;
                perspective = lane % 2;
                source = this.counterfactualSource[lane] != null ?
                    this.counterfactualSource[lane] : this.selfPlayClients[pair].gs;
                iface = this.selfPlayClients[pair].ais[perspective];
            } else {
                JNIGridnetClient client = this.clients[lane - selfLanes];
                perspective = this.playerIds[lane];
                source = this.counterfactualSource[lane] != null ?
                    this.counterfactualSource[lane] : client.gs;
                iface = client.ai1;
            }
            int other = perspective == 0 ? 1 : 0;
            GameState branch = source.clone();
            PlayerAction paSelf = iface.getAction(perspective, branch, selfActions[lane]);
            PlayerAction paOpp = iface.getAction(other, branch, opponentActions[lane]);
            branch.issueSafe(paSelf);
            branch.issueSafe(paOpp);
            branch.cycle();
            this.counterfactualFullState[lane] = encodeFullState(branch, perspective);
            this.counterfactualFullGlobals[lane] = encodeFullGlobals(branch, perspective);
        }
    }

    public Responses reset(int[] nArray) throws Exception {
        int n;
        for (n = 0; n < this.selfPlayClients.length; ++n) {
            this.selfPlayClients[n].reset();
            this.rs[n * 2] = this.selfPlayClients[n].getResponse(0);
            this.rs[n * 2 + 1] = this.selfPlayClients[n].getResponse(1);
        }
        for (n = this.selfPlayClients.length * 2; n < nArray.length; ++n) {
            // Patched: the wrapper passes zeros; the stored per-lane seat wins.
            this.rs[n] = this.clients[n - this.selfPlayClients.length * 2].reset(this.playerIds[n]);
        }
        for (n = 0; n < this.rs.length; ++n) {
            this.observation[n] = this.rs[n].observation;
            this.reward[n] = this.rs[n].reward;
            this.done[n] = this.rs[n].done;
            this.opponentAction[n] = new int[this.gridCells][7];
        }
        for (n = 0; n < this.selfPlayClients.length; ++n) {
            refreshFullState(n * 2, this.selfPlayClients[n].gs, 0);
            refreshFullState(n * 2 + 1, this.selfPlayClients[n].gs, 1);
        }
        for (n = this.selfPlayClients.length * 2; n < this.rs.length; ++n) {
            JNIGridnetClient client = this.clients[n - this.selfPlayClients.length * 2];
            refreshFullState(n, client.gs, this.playerIds[n]);
        }
        this.responses.set(this.observation, this.reward, this.done);
        return this.responses;
    }

    public Responses gameStep(int[][][] nArray, int[] nArray2) throws Exception {
        int n;
        int n2;
        for (n2 = 0; n2 < this.selfPlayClients.length; ++n2) {
            this.counterfactualSource[n2 * 2] = this.selfPlayClients[n2].gs.clone();
            this.counterfactualSource[n2 * 2 + 1] = this.selfPlayClients[n2].gs.clone();
            this.selfPlayClients[n2].gameStep(nArray[n2 * 2], nArray[n2 * 2 + 1]);
            this.rs[n2 * 2] = this.selfPlayClients[n2].getResponse(0);
            this.rs[n2 * 2 + 1] = this.selfPlayClients[n2].getResponse(1);
            int n3 = n2 * 2;
            this.envSteps[n3] = this.envSteps[n3] + 1;
            int n4 = n2 * 2 + 1;
            this.envSteps[n4] = this.envSteps[n4] + 1;
            if (!this.rs[n2 * 2].done[0] && this.envSteps[n2 * 2] < this.maxSteps) continue;
            for (n = 0; n < this.terminalReward1.length; ++n) {
                this.terminalReward1[n] = this.rs[n2 * 2].reward[n];
                this.terminalRone1[n] = this.rs[n2 * 2].done[n];
                this.terminalReward2[n] = this.rs[n2 * 2 + 1].reward[n];
                this.terminalRone2[n] = this.rs[n2 * 2 + 1].done[n];
            }
            this.terminalObservation[n2 * 2] = copyObs(this.rs[n2 * 2].observation);
            this.terminalObservation[n2 * 2 + 1] = copyObs(this.rs[n2 * 2 + 1].observation);
            this.terminalFullState[n2 * 2] = encodeFullState(this.selfPlayClients[n2].gs, 0);
            this.terminalFullGlobals[n2 * 2] = encodeFullGlobals(this.selfPlayClients[n2].gs, 0);
            this.terminalFullState[n2 * 2 + 1] = encodeFullState(this.selfPlayClients[n2].gs, 1);
            this.terminalFullGlobals[n2 * 2 + 1] = encodeFullGlobals(this.selfPlayClients[n2].gs, 1);
            this.selfPlayClients[n2].reset();
            for (n = 0; n < this.terminalReward1.length; ++n) {
                this.rs[n2 * 2].reward[n] = this.terminalReward1[n];
                this.rs[n2 * 2].done[n] = this.terminalRone1[n];
                this.rs[n2 * 2 + 1].reward[n] = this.terminalReward2[n];
                this.rs[n2 * 2 + 1].done[n] = this.terminalRone2[n];
            }
            this.rs[n2 * 2].done[0] = true;
            this.rs[n2 * 2 + 1].done[0] = true;
            this.envSteps[n2 * 2] = 0;
            this.envSteps[n2 * 2 + 1] = 0;
        }
        for (n2 = this.selfPlayClients.length * 2; n2 < nArray2.length; ++n2) {
            int n5 = n2;
            this.envSteps[n5] = this.envSteps[n5] + 1;
            JNIGridnetClient client = this.clients[n2 - this.selfPlayClients.length * 2];
            this.counterfactualSource[n2] = client.gs.clone();
            // Patched: per-lane seat instead of the wrapper's hardcoded zeros.
            this.rs[n2] = client.gameStep(nArray[n2], this.playerIds[n2]);
            // Patched: surface the scripted bot's action for this same tick.
            this.opponentAction[n2] = encodePlayerAction(client.pa2);
            if (!this.rs[n2].done[0] && this.envSteps[n2] < this.maxSteps) continue;
            for (n = 0; n < this.rs[n2].reward.length; ++n) {
                this.terminalReward1[n] = this.rs[n2].reward[n];
                this.terminalRone1[n] = this.rs[n2].done[n];
            }
            this.terminalObservation[n2] = copyObs(this.rs[n2].observation);
            this.terminalFullState[n2] = encodeFullState(client.gs, this.playerIds[n2]);
            this.terminalFullGlobals[n2] = encodeFullGlobals(client.gs, this.playerIds[n2]);
            client.reset(this.playerIds[n2]);
            for (n = 0; n < this.rs[n2].reward.length; ++n) {
                this.rs[n2].reward[n] = this.terminalReward1[n];
                this.rs[n2].done[n] = this.terminalRone1[n];
            }
            this.rs[n2].done[0] = true;
            this.envSteps[n2] = 0;
        }
        for (n2 = 0; n2 < this.rs.length; ++n2) {
            this.observation[n2] = this.rs[n2].observation;
            this.reward[n2] = this.rs[n2].reward;
            this.done[n2] = this.rs[n2].done;
        }
        for (n2 = 0; n2 < this.selfPlayClients.length; ++n2) {
            refreshFullState(n2 * 2, this.selfPlayClients[n2].gs, 0);
            refreshFullState(n2 * 2 + 1, this.selfPlayClients[n2].gs, 1);
        }
        for (n2 = this.selfPlayClients.length * 2; n2 < this.rs.length; ++n2) {
            JNIGridnetClient client = this.clients[n2 - this.selfPlayClients.length * 2];
            refreshFullState(n2, client.gs, this.playerIds[n2]);
        }
        this.responses.set(this.observation, this.reward, this.done);
        return this.responses;
    }

    public int[][][][] getMasks(int n) throws Exception {
        int n2;
        for (n2 = 0; n2 < this.selfPlayClients.length; ++n2) {
            this.masks[n2 * 2] = this.selfPlayClients[n2].getMasks(0);
            this.masks[n2 * 2 + 1] = this.selfPlayClients[n2].getMasks(1);
        }
        for (n2 = this.selfPlayClients.length * 2; n2 < this.masks.length; ++n2) {
            // Patched: mask for the lane's actual Python seat, not the passed 0.
            this.masks[n2] = this.clients[n2 - this.selfPlayClients.length * 2].getMasks(this.playerIds[n2]);
        }
        return this.masks;
    }

    public void close() throws Exception {
        for (JNIGridnetClient object : this.clients) {
            object.close();
        }
        for (JNIGridnetClientSelfPlay jNIGridnetClientSelfPlay : this.selfPlayClients) {
            jNIGridnetClientSelfPlay.close();
        }
    }

    public class Responses {
        public int[][][][] observation;
        public double[][] reward;
        public boolean[][] done;

        public Responses(int[][][][] nArray, double[][] dArray, boolean[][] blArray) {
            this.observation = nArray;
            this.reward = dArray;
            this.done = blArray;
        }

        public void set(int[][][][] nArray, double[][] dArray, boolean[][] blArray) {
            this.observation = nArray;
            this.reward = dArray;
            this.done = blArray;
        }
    }
}
