'''
data : 768 * 128 768?128???
P : 2
init_centroids : ??????  2 * 256 * 64
max_iter : ??????? ??????
codebooks : 2 * 256 * 64
codes : 768 * 2 (768 ? ??64???)
'''
import numpy as np
from scipy.spatial.distance import cdist

def get_index(lst=None, item='', data=None):
    return [data[index] for (index,value) in enumerate(lst) if value == item]

def find_cluster_by_manhattan(data, init_centroids):
    dists = cdist(data, init_centroids, metric='cityblock')
    return np.argmin(dists, axis = 1).tolist()

def k_medians(data, init_centroids, max_iter):
    for iter_num in range(0, max_iter):
        # put every data node into 256 clusters     item : 64 dimension
        category_dict = {}
        clusters = find_cluster_by_manhattan(data, init_centroids)
        
        keys = list(set(clusters))
        keys.sort(key=clusters.index)
        group_values = [get_index(clusters, key, data) for key in keys]
        category_dict = dict(zip(keys, group_values)) 
        
        # calculate the new centroid
        for category in category_dict.keys():
            new = np.median(category_dict[category], axis=0)
            init_centroids[category] = new
    
    # calculate the code, the part of code
    code = find_cluster_by_manhattan(data, init_centroids)
    return init_centroids, code

def pq(data, P, init_centroids, max_iter):
    # split data into P section, if P == 2, then every section is 768 * 64, training seperately
    part = 0
    codes = []
    codebooks = []
    
    for p_part_data in np.split(data, P, axis = 1):
        codebook, code = k_medians(p_part_data, init_centroids[part].copy(), max_iter)
        
        # combine the whole code and codebook
        codebooks.append(codebook)
        codes.append(code)
        part = part + 1
        
    return np.array(codebooks), np.array(codes).transpose().astype(np.uint8)
'''
queries : Q * M, Q is the number of query vectors, M is the dimensionality.
T : the minimum number of returned candidates for each query
codebooks: 768 * 2
codes: 2 * 256 * 64
queries : 1 * 128, if P = 2, the spilt into two 1 * 64 
'''

'''
sort and according to distance transfer to index
'''
class minHeap():
    def __init__(self):
        self.pairs = []
        self.values = []
        self.already_visit = []

    def push(self, value, pair):
        if pair not in self.pairs and pair not in self.already_visit:
            self.values.append(value)
            self.values.sort()
            position = self.values.index(value)
            self.pairs.insert(position, pair)

    def pop(self):
        pair = self.pairs.pop(0)
        del self.values[0]
        self.already_visit.append(pair)
        return pair

def cg_ad_value_by_pos(lists, i, val):
    lists[i, :] = val
    return [list(set(item)) for item in lists]
    
def index_to_point(index_ls, total_index_ls, P):
    return [total_index_ls[i][index_ls[i]] for i in range(0, P)]

def index_to_value(index_ls, total_index_ls, total_dist_ls, P):
    return sum([total_dist_ls[i][index_to_point(index_ls, total_index_ls, P)[i]] for i in range(0, P)])
    
def part_query_dists(data, codebooks, codes):
    # filter codebook, generate new codes and autual points in axis
    new_codebooks = codebooks
    new_codes = codes
    # calculate the distance between query and codebook in this dimension
    dists = cdist(data, new_codebooks, metric='cityblock')
    # sort the distance, know which index is smaller
    index_sort = np.argsort(dists)[0]
    
    # p_axis is visual axis from (0, 0, 0....)
    p_axis = [np.where(index_sort == code)[0][0] for code in new_codes]
    
    return index_sort, dists[0], new_codes, p_axis

def query(queries, codebooks, codes, T):
    # translate codes into a dict
    return [part_query(query.reshape(-1, queries.shape[1]), codebooks.copy(), codes, T) for query in queries]

def part_query(queries, codebooks, codes, T):
    if T <= 0:
        return []
    P = codebooks.shape[0]
    part = 0
    
    heap = minHeap()
    total_index_ls = []
    total_dist_ls = []
    
    candidate = set()
    new_codes = []
    codes_asix = []
    
    # seperate into P part
    for part_queries in np.split(queries, P, axis = 1):
        index_ls, dist_ls, new_part_code, code_asix = part_query_dists(part_queries, codebooks[part].copy(), codes.transpose()[part].copy())
        total_index_ls.append(index_ls)
        total_dist_ls.append(dist_ls)
        new_codes.append(new_part_code)
        codes_asix.append(code_asix)
        
        part += 1
    
    new_codes = np.asarray(new_codes).transpose()
    codes_asix = np.asarray(codes_asix).transpose()
    codes_asix_transpose = codes_asix.transpose()
    
    # put initial node(0, 0....) and  into min heap 
    init_i = [0 for i in range(0, P)]
    init_v = index_to_value(init_i, total_index_ls, total_dist_ls, P)
    heap.push(init_v, init_i)
    
    all_zero_ls = [list(code) for code in codes_asix if (code == np.asarray(init_i)).any()]
 
    all_zero_ls_v = [index_to_value(i, total_index_ls, total_dist_ls, P) for i in all_zero_ls]
    push_event = [heap.push(all_zero_ls_v[i], all_zero_ls[i]) for i in range(0, len(all_zero_ls))]
    
    while len(candidate) < T:
        pair = heap.pop()
        v = [i + 1 for i in pair]
        # find nearest point in P dimension in new codes, put into heap
        
        nearest_i = [list(code) for code in codes_asix if (code == np.asarray(v)).any()]  # error
        
        nearest_v = [index_to_value(i, total_index_ls, total_dist_ls, P) for i in nearest_i]
        event = [heap.push(nearest_v[i], nearest_i[i]) for i in range(0, len(nearest_i))]
        
        nearest_p = index_to_point(pair, total_index_ls, P)
        
        # find pointer in cluster in codes, not new_codes
        cluster = [x for (x, y) in enumerate(cdist([nearest_p], new_codes, metric='cityblock')[0]) if y == 0]
        
        candidate = candidate | set(cluster)
                 
    return candidate