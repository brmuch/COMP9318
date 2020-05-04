## import modules here 

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    for i in range(0, x): 
        if i * i > x:
            return i - 1
    return x


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    x_new = 0
    for i in range(0, MAX_ITER):
        x_new = x_0 - f(x_0) / fprime(x_0)
        if abs(x_new - x_0) < EPSILON:
            return x_new
        x_0 = x_new
    return x_new


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def make_tree(tokens): # do not change the heading of the function
    if len(tokens) == 0:
        return None
    elif len(tokens) == 1:
        return Tree(tokens[0])
    
    cur_tt = Tree(tokens[0])
    
    sep_tokens = seperate(tokens[2:-1])
    
    for sep_token in sep_tokens:
        res_tt = make_tree(sep_token)
        cur_tt.add_child(res_tt)
        
    return cur_tt       
    
def seperate(words):
    if len(words) == 0:                    # recursive base
        return []
    elif len(words) == 1:
        return [words]
    
    if words[1] != '[':                    # recursive body
        res = seperate(words[1::])
        res.insert(0, [words[0]])
        return res
    else:
        list = []
        brackets = 1
        list.extend(words[0:2])
        
        for i in range(2, len(words)):
            if words[i] == '[':
                brackets += 1
            elif words[i] == ']':
                brackets -= 1
            list.append(words[i])
            
            if brackets == 0:
                break
        
        res = seperate(words[len(list):])
        res.insert(0, list)
        return res  

def max_depth(root): # do not change the heading of the function
    if len(root.children) == 0:
        return 1
    
    max = 0
    for child in root.children:
        depth = max_depth(child)
        max = max if max > depth else depth
    return max + 1