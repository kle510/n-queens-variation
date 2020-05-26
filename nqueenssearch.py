import numpy as np
import copy
import math
import random
import time


# generate children for a state
def generate_children(nursery, depth):

    children = []
    starting_pos = 0
    tree_count = 0
    tree_locations = []
    total_child_combos = 1

    # grab row to observe
    row = nursery[depth]
    # one scan of the array to get tree count & locations
    for index, item in enumerate(row):
        # if a tree is there
        if row[index] == 2:
            tree_count += 1
            tree_locations.append(index)

    while tree_count > 0:
        curr_tree = tree_locations.pop(0)
        # get all child combinations up to tree
        open_space = curr_tree-starting_pos
        child_combos = open_space + 1  # account for all 0's case

        # if not at starting position, copy the list
        if starting_pos != 0:
            children_copy = copy.deepcopy(children)

            # open_space-1 because we don't want to rewrite the original childrens list
            for item in (range(child_combos - 1)):
                children.extend(copy.deepcopy(children_copy))

        # resolve case where tree is blocking a potential opening
        if (open_space == 0) and (starting_pos == 0):
            children.append([])

        for index, item in enumerate(range(starting_pos, curr_tree)):
            sublist = [0]*(open_space)
            sublist[index] = 1
            # make list for for all 0's case
            zerolist = [0] * (open_space)
            # if no children, add directly to list
            if starting_pos == 0:
                if index == (curr_tree - 1):
                    children.append(sublist)
                    children.append(zerolist)
                else:
                    children.append(sublist)
            # elseif children, add to back of children
            else:
                for subindex, subitem in enumerate(range(1, total_child_combos + 1)):
                    # if condition for zero list
                    if index == (curr_tree - starting_pos - 1):
                        children[index * total_child_combos +
                                 subindex] += sublist
                        children[(index+1) * total_child_combos +
                                 subindex] += zerolist
                    else:
                        children[index * total_child_combos +
                                 subindex] += sublist

        # add the tree to each child
        for child in children:
            child.append(2)

        # update variables
        tree_count -= 1
        starting_pos = curr_tree + 1
        total_child_combos *= child_combos

    # finish it (or if there is no trees)
    open_space = len(row) - starting_pos
    child_combos = open_space + 1

    # if not at starting position, copy the list
    if starting_pos != 0:
        children_copy = copy.deepcopy(children)
        # child_combos -1 because we don't want to rewrite the original childrens list
        for item in range(child_combos - 1):
            children.extend(copy.deepcopy(children_copy))
    # resolve case where tree is blocking a potential opening
    if (open_space == 0) and (starting_pos == 0):
        children.append([])

    for index, item in enumerate(range(starting_pos, len(row))):
        sublist = [0] * (open_space)
        sublist[index] = 1
        zerolist = [0] * (open_space)
        # if no children, add directly to list
        if starting_pos == 0:
            if index == (len(row)-1):
                children.append(sublist)
                children.append(zerolist)
            else:
                children.append(sublist)
        # elseif children, add to back of children
        else:
            for subindex, subitem in enumerate(range(1, total_child_combos + 1)):
                # if condition for zero list
                if index == (len(row) - starting_pos - 1):
                    children[index * total_child_combos + subindex] += sublist
                    children[(index + 1) * total_child_combos +
                             subindex] += zerolist
                else:
                    children[index * total_child_combos + subindex] += sublist

    #print ("children: ", children)
    return children

# Check to see if current state is a valid goal state.


def goal_test(state, lizard_num):
    # check to see if all the lizards are on the board
    lizard_count = sum(lizards.count(1) for lizards in state)
    if lizard_count != int(lizard_num):
        return False

    # check rows
    for x, row in enumerate(state):
        lizard_present = False
        for y, index in enumerate(row):
            if state[x][y] == 1:
                if lizard_present == True:
                    return False
                lizard_present = True
            elif state[x][y] == 2:
                lizard_present = False

    # check columns
    for y, col in enumerate(np.transpose(state)):
        lizard_present = False
        for x, index in enumerate(col):
            if state[x][y] == 1:
                if lizard_present == True:
                    return False
                lizard_present = True
            elif state[x][y] == 2:
                lizard_present = False

    # check diagonals
    x_len = len(state[0])
    y_len = len(state)
    state_array = np.array(state)
    diagonals = [state_array.diagonal(i) for i in range(y_len - 1, -x_len, -1)]
    diagonals.extend(state_array[::-1, :].diagonal(i)
                     for i in range(-x_len + 1, y_len))

    diag_matrix = [diag.tolist() for diag in diagonals]

    for i, diag in enumerate(diag_matrix):
        lizard_present = False
        for j, index in enumerate(diag):
            if diag_matrix[i][j] == 1:
                if lizard_present == True:
                    return False
                lizard_present = True
            elif diag_matrix[i][j] == 2:
                lizard_present = False

    return True


def general_search(algo, dim, lizard_num, nursery):

    nodes = []
    state = nursery
    parent = None
    depth = 0
    children = generate_children(nursery, depth)

    nodes.append({'state': state, 'parent': parent,
                  'children': children, 'depth': depth})

    while len(nodes) > 0:

        node = nodes.pop(0)

        #print ("node['state'] ", node['state'])

        if goal_test(node['state'], lizard_num):
            return node

        elif node['children'] != None:

            for index, child in enumerate(node['children']):

                new_parent = node
                new_state = copy.deepcopy(node['state'])
                new_state[int(node['depth'])] = node['children'][index]

                if node['depth'] + 1 == int(dim):
                    new_children = None
                else:
                    new_children = generate_children(nursery, node['depth']+1)

                new_depth = node['depth'] + 1
                new_node = {'state': new_state, 'parent': new_parent,
                            'children': new_children, 'depth': new_depth}

                if algo == "BFS":
                    nodes.append(new_node)
                elif algo == "DFS":
                    nodes.insert(0, new_node)

    return None


def generate_initial_state(dim, lizard_num, nursery):

    initial_state = nursery
    lizard_loc = []

    while lizard_num > 0:
        row = random.randint(0, dim - 1)
        col = random.randint(0, dim - 1)
        if initial_state[row][col] == 0:
            initial_state[row][col] = 1
            lizard_loc.append([row, col])
            lizard_num -= 1

    return [initial_state, lizard_loc]


def simulated_annealing(initial_state_lizard_loc, dim, lizard_num):

    current_temp = 100.0
    alpha = 0.99992
    final_temp = 0.00005
    current_state = initial_state_lizard_loc[0]
    current_lizard_loc = initial_state_lizard_loc[1]
    start_SA_time = time.time()

    while current_temp > 0:

        current_temp = current_temp * alpha

        curr_SA_time = time.time()
        SA_time = curr_SA_time - start_SA_time
        if (SA_time) > 295.0:
            return None

        if (current_temp < final_temp):
            if goal_test(current_state, lizard_num):
                return current_state

            else:
                return None
        next_state_lizard_loc = generate_neighbor(
            current_state, current_lizard_loc, dim, lizard_num)
        next_state = next_state_lizard_loc[0]
        next_lizard_loc = next_state_lizard_loc[1]

        # want to minimize energy i.e. get a negative number
        next_cost = cost(next_state)
        current_cost = cost(current_state)

        energy = next_cost - current_cost
        #print("energy: ", next_cost, "-",current_cost,"=",energy)

        if current_cost == 0:
            if goal_test(current_state, lizard_num):
                return current_state
            else:
                return None

        if energy < 0:
            current_state = next_state
            current_lizard_loc = next_lizard_loc
        else:
            probability = math.exp(-energy/current_temp)
            random_var = random.random()

            if probability > random_var:
                current_state = next_state
                current_lizard_loc = next_lizard_loc

    return None


def cost(state):

    energy_cost = 0

    # check rows
    for x, row in enumerate(state):
        lizard_present = False
        for y, index in enumerate(row):
            if state[x][y] == 1:
                if lizard_present == True:
                    energy_cost += 1
                lizard_present = True
            elif state[x][y] == 2:
                lizard_present = False

    # check columns
    for y, col in enumerate(np.transpose(state)):
        lizard_present = False
        for x, index in enumerate(col):
            if state[x][y] == 1:
                if lizard_present == True:
                    energy_cost += 1
                lizard_present = True
            elif state[x][y] == 2:
                lizard_present = False

    # check diagonals
    x_len = len(state[0])
    y_len = len(state)
    state_array = np.array(state)
    diagonals = [state_array.diagonal(i) for i in range(y_len - 1, -x_len, -1)]
    diagonals.extend(state_array[::-1, :].diagonal(i)
                     for i in range(-x_len + 1, y_len))
    diag_matrix = [diag.tolist() for diag in diagonals]

    for i, diag in enumerate(diag_matrix):
        lizard_present = False
        for j, index in enumerate(diag):
            if diag_matrix[i][j] == 1:
                if lizard_present == True:
                    energy_cost += 1
                lizard_present = True
            elif diag_matrix[i][j] == 2:
                lizard_present = False

    return energy_cost


def generate_neighbor(current_state, current_lizard_loc, dim, lizard_num):

    next_state = copy.deepcopy(current_state)
    next_lizard_loc = copy.deepcopy(current_lizard_loc)

    # pick random lizard
    random_lizard = random.randint(0, lizard_num - 1)

    # set old random lizard position to 0
    old_row = next_lizard_loc[random_lizard][0]
    old_col = next_lizard_loc[random_lizard][1]
    next_state[old_row][old_col] = 0

    place_random_lizard = False

    while place_random_lizard == False:
        new_row = random.randint(0, dim - 1)
        new_col = random.randint(0, dim - 1)

        if next_state[new_row][new_col] == 0:
            next_state[new_row][new_col] = 1
            next_lizard_loc[random_lizard] = [new_row, new_col]
            place_random_lizard = True

    return [next_state, next_lizard_loc]


def main():

    input = open("input.txt", "r")

    output = open("output.txt", "w")

    algo = input.readline().strip()
    dim = input.readline().strip()
    lizard_num = input.readline().strip()
    nursery = []
    row_scanner = input.readline().strip()

    while row_scanner:
        currRow = [[int(i) for i in row_scanner]]
        nursery += currRow
        row_scanner = input.readline().strip()

    if algo == "BFS":
        result = general_search(algo, dim, lizard_num, nursery)

    elif algo == "DFS":
        result = general_search(algo, dim, lizard_num, nursery)
    else:
        initial_state_lizard_loc = generate_initial_state(
            int(dim), int(lizard_num), nursery)
        start_SA_time = time.time()
        result = simulated_annealing(
            initial_state_lizard_loc, int(dim), int(lizard_num))

    if result == None:
        output.write("FAIL")
    else:
        output.write("OK" + "\n")

        if algo == "SA":
            solution = result
        else:
            solution = result['state']

        for index, row in enumerate(solution):
            if index == len(solution) - 1:
                output.write(''.join(str(i) for i in row))
            else:
                output.write(''.join(str(i) for i in row) + "\n")

    input.close()
    output.close()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start, " seconds")
