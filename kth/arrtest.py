import numpy as np

x = []
num = 3
for i in range(num):
	y = []
	for j in range(num+1):
		z = []
		for k in range(num+8):
			w = []
			for l in range(num+300):
				v = []	
				for m in range(num+500):
					v.append(i)
					v.append(j)
					v.append(k)
					v.append(l)
					v.append(m)
				w.append(v)
			z.append(w)
		y.append(z)	
	x.append(y)
for i in range(num):
	print("x[", i, "]=", x[i], "\n")
print("length=", len(x), "\n")
x = np.asarray(x)
for i in range(num):
	print("shape=", x[i].shape, "\n")
print("shape=", x.shape, "\n")
