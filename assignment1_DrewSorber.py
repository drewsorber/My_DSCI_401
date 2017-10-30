import itertools

#Flatten
def flatten(l):
	flatlist= []
	for val in l:
		if(type(val) == list): 
			flatlist.extend(flatten(val))
		else:
			flatlist.append(val)
	return flatlist

#Powerset
def powerset(l):
    result = [[]]
    for val in l:
        result.extend([subset + [val] for subset in result])
    return result


#Permutation
def permutation(l):
	return list(itertools.permutations(l))

#Number Spiral 
#Not 100% working
def spiral(n, end_corner):
	w, h = n, n
	square = [[0 for x in range(w)] for y in range(h)]
	x = n/2
	y = n/2
	direc = 3
	if(end_corner == 1):
		direc = direc - 1
		y = y - 1
	if(end_corner == 3):
		direc = direc - 2
		x = x - 1
		y = y - 1
	if(end_corner == 4):
		direc = direc - 3
		x = x - 1
 	val = 0
	for i in range(n):
		for j in range(2):
			h = i
			if j == 1:
				h = 1 + h
			for k in range(h):
				square[x][y] = val
				val = val + 1
				if (direc == 0):
					x = x - 1
				if (direc == 1):
					y = y - 1
				if (direc == 2):
					x = x + 1
				if (direc == 3):
					y = y + 1
			direc = direc + 1
			if (direc == 4):
				direc = 0
	for l in range(n):
		for m in range(n):
			print(square[m][l])
		print('\n')