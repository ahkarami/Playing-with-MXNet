# Besmei Taala
# Amir Hossein Karami
# Subject: Learn & Work with MXNet Deep Learning Framework (Gluon)
# Main Link: https://gluon-crash-course.mxnet.io/index.html


from mxnet import nd


a1 = nd.array(((1, 2, 3), (5, 6, 7)))
print(a1)

a2 = nd.ones((2, 3))
print(a2)

a3 = nd.random.uniform(0, 1, (2, 3))  # min_value, max_value, shape
print(a3)

a4 = nd.full((2, 3), 2.0)  # shape, full matrix with value
print(a4)

print(a4.shape, a4.size, a4.dtype)


# *** Section2 (Operations):
a5 = a3 * a4  # element-wise multiplication
print(a5)

a6 = nd.dot(a5, a2.T)
print(a6)


# up to: https://gluon-crash-course.mxnet.io/ndarray.html#Indexing

