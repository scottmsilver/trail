// Native A* kernel for engine v2's terrain-aware pathfinder.
//
// A faithful port of TerrainAwarePathfinder.find_path(): identical cost math
// (libm sqrt/atan2/pow, same operation order), identical neighbor order,
// identical best-g pruning, and -- critically -- a byte-for-byte port of
// CPython's heapq _siftdown/_siftup so the open-set pop order (and thus the
// path chosen among equal-f_cost ties) is identical to the pure-Python engine.
//
// C++ is used purely for std::vector (RAII: no manual malloc/free/realloc).
// We deliberately do NOT use std::priority_queue -- its heap layout differs
// from heapq and would change tie-breaking, breaking bit-identical output.
//
// Compiled on demand with g++ via ctypes (see pathfinder_native.py). The
// entry point is exported with C linkage so ctypes can bind `astar`.
#include <cmath>
#include <vector>

// PathType ints (must match app/engine_v2/path_layer.py PathType IntEnum).
namespace {
constexpr int PT_OBSTACLE = 1;
constexpr int PT_TRAIL = 2;
constexpr int PT_PATH = 3;
constexpr int PT_NATURAL = 6;

// One entry per push (mirrors Python creating a TerrainNode).
struct Node {
    int cell;    // row*cols + col
    int parent;  // parent node id, -1 for start
    double g;
    double f;
    double steep;  // consecutive_steep_distance
};

// Euclidean grid distance with deltas widened to double before squaring, so
// the intermediate product cannot overflow int.
inline double grid_dist(int r0, int c0, int r1, int c1) {
    double dr = static_cast<double>(r0 - r1);
    double dc = static_cast<double>(c0 - c1);
    return std::sqrt(dr * dr + dc * dc);
}

// --- CPython heapq port. Heap stores node ids; order key is pool[id].f. ---
inline bool heap_lt(const std::vector<Node>& pool, int a, int b) {
    return pool[a].f < pool[b].f;
}

void siftdown(std::vector<int>& heap, const std::vector<Node>& pool, int startpos, int pos) {
    int newitem = heap[pos];
    while (pos > startpos) {
        int parentpos = (pos - 1) >> 1;
        int parent = heap[parentpos];
        if (heap_lt(pool, newitem, parent)) {
            heap[pos] = parent;
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

void siftup(std::vector<int>& heap, const std::vector<Node>& pool, int endpos, int pos) {
    int startpos = pos;
    int newitem = heap[pos];
    int childpos = 2 * pos + 1;
    while (childpos < endpos) {
        int rightpos = childpos + 1;
        if (rightpos < endpos && !heap_lt(pool, heap[childpos], heap[rightpos])) {
            childpos = rightpos;
        }
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    siftdown(heap, pool, startpos, pos);
}
}  // namespace

// Returns path length (cells written to out_path, start..end), 0 if no path,
// -1 on invalid dimensions. out_nodes_explored gets the pop count. out_path
// must hold at least rows*cols ints.
extern "C" int astar(
    const double* elevation, const int* terrain,
    int rows, int cols,
    int start_idx, int end_idx,
    double resolution, double heuristic_weight,
    double elevation_weight, double elevation_exponent,
    double max_slope_degrees, double steep_threshold,
    double fatigue_distance, double fatigue_exponent,
    double sustained_slope_weight,
    const double* terrain_cost, int ncost,  // cost[t]; NaN => use 1.0
    int* out_path, int* out_nodes_explored) {
    // Guard against index/pool-counter overflow: bound rows*cols so flat
    // indices and up to ~8 pushes/cell stay within int. Real DEM tiles are far
    // below this.
    if (rows <= 0 || cols <= 0 || static_cast<long>(rows) * cols > (2147483647L / 8)) return -1;

    const long ncells = static_cast<long>(rows) * cols;
    const double RAD2DEG = 180.0 / M_PI;
    const int end_row = end_idx / cols;
    const int end_col = end_idx % cols;
    const int start_row = start_idx / cols;
    const int start_col = start_idx % cols;

    double straight = resolution * grid_dist(start_row, start_col, end_row, end_col);
    double sld = straight > 1.0 ? straight : 1.0;

    // 8 neighbor offsets in get_neighbors() order + per-step horiz distance.
    int off_dr[8], off_dc[8];
    double off_h[8];
    int no = 0;
    for (int dr = -1; dr <= 1; dr++)
        for (int dc = -1; dc <= 1; dc++) {
            if (dr == 0 && dc == 0) continue;
            off_dr[no] = dr;
            off_dc[no] = dc;
            off_h[no] = resolution * std::sqrt(static_cast<double>(dr * dr + dc * dc));
            no++;
        }

    std::vector<double> best_g(ncells, INFINITY);
    std::vector<char> closed(ncells, 0);
    std::vector<Node> pool;
    std::vector<int> heap;
    pool.reserve(ncells);
    heap.reserve(ncells);

    // start node
    double sh = heuristic_weight * (resolution * grid_dist(start_row, start_col, end_row, end_col));
    pool.push_back(Node{start_idx, -1, 0.0, 0.0 + sh, 0.0});
    heap.push_back(0);
    siftdown(heap, pool, 0, 0);
    best_g[start_idx] = 0.0;

    int nodes_explored = 0;
    int goal_node = -1;
    int heaplen = 1;

    while (heaplen > 0) {
        // heappop
        int lastelt = heap[heaplen - 1];
        heap.pop_back();
        heaplen--;
        int cur;
        if (heaplen > 0) {
            cur = heap[0];
            heap[0] = lastelt;
            siftup(heap, pool, heaplen, 0);
        } else {
            cur = lastelt;
        }

        int ccell = pool[cur].cell;
        if (closed[ccell]) continue;

        nodes_explored++;

        if (ccell == end_idx) {
            goal_node = cur;
            break;
        }

        closed[ccell] = 1;

        int cr = ccell / cols, cc = ccell % cols;
        double elev_from = elevation[ccell];
        int terrain_from = terrain[ccell];
        double g_from = pool[cur].g;
        double steep_from = pool[cur].steep;
        double dev_over = g_from / sld - 1.5;
        if (dev_over < 0.0) dev_over = 0.0;
        double deviation_penalty = 0.1 * (dev_over * dev_over);  // max(0,x)**2
        bool from_is_trail = (terrain_from == PT_TRAIL);

        for (int k = 0; k < no; k++) {
            int next_row = cr + off_dr[k];
            int next_col = cc + off_dc[k];
            if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) continue;
            int nidx = next_row * cols + next_col;
            int terrain_to = terrain[nidx];
            if (terrain_to == PT_OBSTACLE) continue;
            if (closed[nidx]) continue;

            double horiz_distance = off_h[k];
            double elev_to = elevation[nidx];
            double elevation_change = std::fabs(elev_to - elev_from);
            double slope_degrees = std::atan2(elevation_change, horiz_distance) * RAD2DEG;
            if (slope_degrees > max_slope_degrees) continue;

            double terrain_multiplier = 1.0;
            if (terrain_to >= 0 && terrain_to < ncost && !std::isnan(terrain_cost[terrain_to]))
                terrain_multiplier = terrain_cost[terrain_to];
            if (from_is_trail && !(terrain_to == PT_TRAIL || terrain_to == PT_PATH || terrain_to == PT_NATURAL))
                terrain_multiplier *= 1.5;

            double elevation_penalty = elevation_weight * std::pow(slope_degrees / 10.0, elevation_exponent);
            if (elev_to > elev_from) elevation_penalty *= 1.5;

            double new_steep_distance, sustained_penalty;
            if (slope_degrees > steep_threshold) {
                new_steep_distance = steep_from + horiz_distance;
                double fatigue_factor = std::pow(new_steep_distance / fatigue_distance, fatigue_exponent);
                sustained_penalty = sustained_slope_weight * fatigue_factor * horiz_distance;
            } else {
                new_steep_distance = steep_from - horiz_distance * 0.5;
                if (new_steep_distance < 0.0) new_steep_distance = 0.0;
                sustained_penalty = 0.0;
            }

            double move_cost = horiz_distance * terrain_multiplier *
                               (1.0 + elevation_penalty + sustained_penalty + deviation_penalty);
            double new_g_cost = g_from + move_cost;

            if (new_g_cost >= best_g[nidx]) continue;  // INF sentinel => absent
            best_g[nidx] = new_g_cost;

            double h = heuristic_weight * (resolution * grid_dist(next_row, next_col, end_row, end_col));
            pool.push_back(Node{nidx, cur, new_g_cost, new_g_cost + h, new_steep_distance});
            heap.push_back(static_cast<int>(pool.size() - 1));
            siftdown(heap, pool, 0, heaplen);
            heaplen++;
        }
    }

    int len = 0;
    if (goal_node >= 0) {
        int cnt = 0;
        for (int m = goal_node; m != -1; m = pool[m].parent) cnt++;
        len = cnt;
        int i = cnt - 1;
        for (int m = goal_node; m != -1; m = pool[m].parent) out_path[i--] = pool[m].cell;
    }
    *out_nodes_explored = nodes_explored;
    return len;
}
