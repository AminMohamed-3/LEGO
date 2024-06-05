class TreeNode:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)


class Tree:
    def __init__(self, root_value):
        self.root = TreeNode(root_value)
        self.nodes = {root_value: self.root}

    def add_parent(self, child_value, parent_value):
        if child_value not in self.nodes:
            self.nodes[child_value] = TreeNode(child_value)
        if parent_value not in self.nodes:
            self.nodes[parent_value] = TreeNode(parent_value)

        child_node = self.nodes[child_value]
        parent_node = self.nodes[parent_value]

        parent_node.add_child(child_node)

    def add_child(self, parent_value, child_value):
        self.add_parent(child_value, parent_value)

    def find_shortest_path(self, start_value, end_value):
        if start_value not in self.nodes or end_value not in self.nodes:
            return -1

        start_node = self.nodes[start_value]
        end_node = self.nodes[end_value]

        def bfs(start, target):
            queue = [(start, 0)]
            visited = set()
            while queue:
                current, depth = queue.pop(0)
                if current == target:
                    return depth
                visited.add(current)
                neighbors = current.children + (
                    [current.parent] if current.parent else []
                )
                for neighbor in neighbors:
                    if neighbor and neighbor not in visited:
                        queue.append((neighbor, depth + 1))
            return -1

        return bfs(start_node, end_node)


# Tree cosntruction
admiration = TreeNode("admiration")
amusement = TreeNode("amusement")
anger = TreeNode("anger")
annoyance = TreeNode("annoyance")
approval = TreeNode("approval")
caring = TreeNode("caring")
confusion = TreeNode("confusion")
curiosity = TreeNode("curiosity")
desire = TreeNode("desire")
disappointment = TreeNode("disappointment")
disapproval = TreeNode("disapproval")
disgust = TreeNode("disgust")
embarrassment = TreeNode("embarrassment")
excitement = TreeNode("excitement")
fear = TreeNode("fear")
gratitude = TreeNode("gratitude")
grief = TreeNode("grief")
joy = TreeNode("joy")
love = TreeNode("love")
nervousness = TreeNode("nervousness")
optimism = TreeNode("optimism")
pride = TreeNode("pride")
realization = TreeNode("realization")
relief = TreeNode("relief")
remorse = TreeNode("remorse")
sadness = TreeNode("sadness")
surprise = TreeNode("surprise")
neutral = TreeNode("neutral")

# level1
ex_joy = TreeNode("ex_joy")
ex_joy.add_child(excitement)
ex_joy.add_child(joy)

des_opt = TreeNode("des_opt")
des_opt.add_child(desire)
des_opt.add_child(optimism)

pride_adm = TreeNode("pride_adm")
pride_adm.add_child(pride)
pride_adm.add_child(admiration)

grat_relief = TreeNode("grat_relief")
grat_relief.add_child(gratitude)
grat_relief.add_child(relief)

app_real = TreeNode("app_real")
app_real.add_child(approval)
app_real.add_child(realization)

cur_conf = TreeNode("cur_conf")
cur_conf.add_child(curiosity)
cur_conf.add_child(confusion)

fear_nerv = TreeNode("fear_nerv")
fear_nerv.add_child(fear)
fear_nerv.add_child(nervousness)

rem_embar = TreeNode("rem_embar")
rem_embar.add_child(remorse)
rem_embar.add_child(embarrassment)

disap_sad = TreeNode("disap_sad")
disap_sad.add_child(disappointment)
disap_sad.add_child(sadness)

ang_ann = TreeNode("ang_ann")
ang_ann.add_child(anger)
ang_ann.add_child(annoyance)

# level2
ex_joy_love = TreeNode("ex_joy_love")
ex_joy_love.add_child(ex_joy)
ex_joy_love.add_child(love)

des_opt_caring = TreeNode("des_opt_caring")
des_opt_caring.add_child(des_opt)
des_opt_caring.add_child(caring)

prid_adm_grat_relief = TreeNode("prid_adm_grat_relief")
prid_adm_grat_relief.add_child(pride_adm)
prid_adm_grat_relief.add_child(grat_relief)

surp_cur_conf = TreeNode("surp_cur_conf")
surp_cur_conf.add_child(surprise)
surp_cur_conf.add_child(cur_conf)

disp_sad_grief = TreeNode("disp_sad_grief")
disp_sad_grief.add_child(disap_sad)
disp_sad_grief.add_child(grief)

disgust_ang_ann = TreeNode("disgust_ang_ann")
disgust_ang_ann.add_child(disgust)
disgust_ang_ann.add_child(ang_ann)

# higher levels
amus_ex_joy_love = TreeNode("amus_ex_joy_love")
amus_ex_joy_love.add_child(amusement)
amus_ex_joy_love.add_child(ex_joy_love)

prid_adm_grat_relief_app_real = TreeNode("prid_adm_grat_relief_app_real")
prid_adm_grat_relief_app_real.add_child(prid_adm_grat_relief)
prid_adm_grat_relief_app_real.add_child(app_real)

prepositive = TreeNode("prepositive")
prepositive.add_child(des_opt_caring)
prepositive.add_child(prid_adm_grat_relief_app_real)

positive = TreeNode("positive")
positive.add_child(amus_ex_joy_love)
positive.add_child(prepositive)

positive_amb = TreeNode("positive_amb")
positive_amb.add_child(positive)
positive_amb.add_child(surp_cur_conf)

rem_embar_disp_sad_grief = TreeNode("rem_embar_disp_sad_grief")
rem_embar_disp_sad_grief.add_child(rem_embar)
rem_embar_disp_sad_grief.add_child(disp_sad_grief)

fear_nerv_rem_embar_disp_sad_grief = TreeNode("fear_nerv_rem_embar_disp_sad_grief")
fear_nerv_rem_embar_disp_sad_grief.add_child(fear_nerv)
fear_nerv_rem_embar_disp_sad_grief.add_child(rem_embar_disp_sad_grief)

disgust_ang_ann_disapproval = TreeNode("disgust_ang_ann_disapproval")
disgust_ang_ann_disapproval.add_child(disgust_ang_ann)
disgust_ang_ann_disapproval.add_child(disapproval)

negative = TreeNode("negative")
negative.add_child(fear_nerv_rem_embar_disp_sad_grief)
negative.add_child(disgust_ang_ann_disapproval)

root = TreeNode("root")
root.add_child(positive_amb)
root.add_child(negative)
root.add_child(neutral)

# Create the Tree instance
emotion_tree = Tree("root")


# Add all the nodes to the Tree
def tree_constructor(node):
    for child in node.children:
        emotion_tree.add_parent(child.value, node.value)
        tree_constructor(child)


tree_constructor(root)


# given two lists of emotions, find the shortest path between the two lists
def get_distance(emotions1, emotions2, verbose=False):

    # make emotions1 the largest of the two
    if len(emotions1) < len(emotions2):
        emotions1, emotions2 = emotions2, emotions1

    for emotion1 in emotions1:
        if emotion1 not in emotion_tree.nodes:
            print(f"{emotion1} not in the tree")
            return 100000

    for emotion2 in emotions2:
        if emotion2 not in emotion_tree.nodes:
            print(f"{emotion2} not in the tree")
            return 100000

    accumulation = 0

    # set for unique paths
    traversed = set()
    for emotion1 in emotions1:
        for emotion2 in emotions2:
            if emotion1 == "neutral" or emotion2 == "neutral":
                path_len = 5
            else:
                path_len = (
                    emotion_tree.find_shortest_path(emotion1, emotion2) - 1
                    if emotion1 != emotion2
                    else 0
                )
            # normalize path length to be between 0 and 1
            if emotion1 == emotion2:
                path_len = -10.4  
            path_len = (path_len - 0) / (13 - 0)
            # if the two emotions have been found before, skip this step
            if (emotion1, emotion2) in traversed or (emotion2, emotion1) in traversed:
                continue
            else:
                traversed.add((emotion1, emotion2))
                traversed.add((emotion2, emotion1))

            if verbose:
                print(f"{emotion1} -> {emotion2}: {path_len}")
            accumulation += path_len
    return accumulation
