import numpy as np;

def learning():
    a = np.array([1,2,3])
    print(a);
    a = np.array([[1,2], [3,4]])
    print(a);
    a = np.array([1, 2, 3, 4, 5], ndmin= 2)
    print(a);
    a = np.array([1, 2, 3], dtype=complex)
    print(a);

    dt = np.dtype(np.int32);
    print(dt);

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a.shape);
    a.shape = (3, 2);
    print(a);
    print(a.reshape(3, 2))
    a = np.arange(24);
    print(a.ndim)
    b = a.reshape(2, 4, 3)
    print(b);

    x = np.array([1, 2, 3, 4, 5], dtype=np.int8);
    print(x.itemsize);
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    print(x.itemsize)
    print(x.flags);

    x = np.empty([3, 2], dtype=int)
    print(x);

    x = np.zeros(5)
    print(x);

    x = np.ones(5)
    print(x);
    print(np.ones([2,2], dtype=int));

    print(np.asarray([1, 2, 3]));
    print(np.asarray([(1, 2, 3), (4, 5)]));

    list = range(5);
    it = iter(list);
    x = np.fromiter(it, dtype=float);
    print(x)

    list = range(5);
    x = np.fromiter(list, dtype=float);
    print(x)

    print(np.arange(10));
    print(np.arange(10, 20, 2))

    print(np.linspace(10, 20, 5));
    print(np.linspace(10, 20, 5, endpoint=False));

    print(np.linspace(1, 2, 5, retstep=True))
    print(np.linspace(1, 2, 5, retstep=False))
    # 指数
    print(np.logspace(1.0, 2.0, num=10))
    print(np.logspace(1, 10, num=10, base=2));

    a = np.arange(0, 10, 2);
    s = slice(2, 5, 2) # the start and the end indicate index.
    print(a);
    print(a[s]);

    print(a[2:5:2]);
    print(a[2]);
    print(a[2:]);

    a = np.array([[1,2,3], [3,4,5],[4,5,6]])
    print(a[1:]);# slice items starting from index
    print(a)
    print(a[..., 1])# this returns array of items in the second column
    print(a[1, ...])# Now we will slice all items from the second row
    print(a[...,1:])# Now we will slice all items from column 1 onwards

    x = np.array([[1, 2], [3, 4], [5, 6]])
    #The selection includes elements at (0,0), (1,1) and (2,0) from the first array.
    y = x[[0, 1, 2], [0, 1, 0]] # copy from x.
    print(y)
    y[0] = 2;
    print(x)

    x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
    print(x);
    rows = np.array([[0, 0], [3, 3]])#
    cols = np.array([[0, 2], [0, 2]])
    #The resultant selection is an ndarray object containing corner elements.
    y = x[rows, cols];
    print(y);

    x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
    print(x);
    z = x[1:4, 1:3];
    print(z);
    print("-----")
    x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
    y = x[1:4, [1, 2]];
    print(y)
    y = x[1:4, 1:3];
    print(y)

    print(x);
    print(x[x> 5])

    a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
    print(a[~np.isnan(a)]) # filter the element which is not none.

    a = np.array([1, 2+6j, 5, 3.5+5j])
    print(a[np.iscomplex(a)])

def broadcasting():
    # broadcasting
    a = np.array([1, 2, 3, 4])
    b = np.array([10, 20, 30, 40])
    c = a * b;
    print(c)
    print(a + b)

    a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]])
    b = np.array([1.0,2.0,3.0])
    print(a)
    print(b)
    print(a + b)
    print(a * b)

    a = np.array([[1, 1], [2, 2]])
    b = np.array([[3, 3], [4, 4]])
    print(a)
    print(b)
    print(a+b)
    print(a * b) # not dot product

    a = np.arange(0, 60, 5);
    a = a.reshape(3, 4);
    print(a)
    b = np.array([1, 2, 3, 4], dtype=int);
    print(b);
    for x, y, in np.nditer([a, b]):
        print('%d:%d' % (x, y), end=' ')


def iterating():
    # iterating over array
    a = np.arange(0, 60, 5);
    a = a.reshape(3, 4)
    print(a)
    for x in np.nditer(a):
        print(x, end=' ')

    #The order of iteration is chosen to match the memory layout of an array, without considering a particular ordering.
    # This can be seen by iterating over the transpose of the above array.
    b = a.T
    print(a)
    print(b)
    for x in np.nditer(b):
        print(x, end=" ")

    a = np.arange(0, 60, 5)
    a = a.reshape(3, 4);
    print(a)
    b = a.T
    print(b);
    c = b.copy(order='C')
    print(c)
    for x in np.nditer(c):
        print(x, end=' ');

    c = b.copy(order='F')
    print(c)
    for x in np.nditer(c):
        print(x, end=' ');

    a = np.arange(0,60,5)
    a = a.reshape(3,4)
    print('Original array is:')
    print(a);

    print('Sorted in C-style order:')
    for x in np.nditer(a, order = 'C'):
       print(x, end=' ')
    print('\n')
    print('Sorted in F-style order:')
    for x in np.nditer(a, order = 'F'):
       print(x, end=' ')
    print('\n')


def modify():
    # modify array
    a = np.arange(0, 60, 5);
    a = a.reshape(3, 4);
    print(a);
    for x in np.nditer(a, op_flags=['readwrite']):
        x[...] = 2*x;
    print(a)

    a = np.arange(0, 60, 5);
    a = a.reshape(3, 4);
    print(a)
    for x in np.nditer(a, flags=['external_loop'], order='C'):
        print(x, end=' ')

    print('\n')
    print(a);
    for x in np.nditer(a, flags=['external_loop'], order='F'):
        print(x, end=' ')


def array_manipulation():
    a = np.arange(0, 60, 5)
    a = a.reshape(3, 4);
    print(a)
    print(a.flatten())
    print(a.flatten(order='F'));

    a = np.arange(8).reshape(2, 4);
    print(a)
    print(a.ravel())
    print(a.ravel(order='F'))

    print(a.T)
    print(a.transpose());

    a = np.arange(4).reshape(1, 4);
    print(a)
    print(np.broadcast_to(a, (4, 4)))

    x = np.array(([1, 2], [3, 4]))
    print(x)
    y = np.expand_dims(x, axis=0)
    print(y)
    print(x.shape, y.shape)

    y = np.expand_dims(x, axis=1)
    print(y)
    print(x.ndim, y.ndim)
    print(x.shape, y.shape)

    x = np.arange(9).reshape(1, 3, 1, 3);
    print(x)
    y = np.squeeze(x); # removes one-dimensional entry
    print(y)
    print(x.shape, y.shape)
    z = np.squeeze(y)
    print(z)
    print(z.shape);

    a = np.array([[1, 2], [3, 4]]);
    print(a)
    b = np.array([[5, 6], [7, 8]])
    print(b)
    # 'Joining the two arrays along axis 0:'
    print(np.concatenate((a,b)))
    # Joining the two arrays along axis 1:
    print(np.concatenate((a, b), axis=1))

    a = np.arange(9)
    print(a)
    b = np.split(a, 3);
    print(b)
    print(np.split(a, [4, 7]))

    a = np.arange(16).reshape(4, 4)
    print(a)
    b = np.hsplit(a, 2);
    print(b)
    c = np.vsplit(a, 2);
    print(c)

    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)
    b = np.resize(a, (3, 2))
    print(b)
    b = np.resize(a, (3, 3))
    print(b)

    print('---')
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a)
    print(np.append(a, [7, 8, 9]))
    print(np.append(a, [[7,8, 9]], axis=0))
    print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1))

    a = np.array([[1, 2], [3, 4], [5, 6]])
    print(a);
    print(np.insert(a, 3, [11, 12]))
    print(np.insert(a, 1, [11], axis=0))
    print(np.insert(a, 1, 11, axis=1))

    a = np.array([5,2,6,2,7,5,6,8,2,9])
    print(np.unique(a))

    u, indeces = np.unique(a, return_index=True)
    print(u, indeces)

    u, indeces = np.unique(a, return_inverse=True)
    print(u, indeces)

    u, indeces = np.unique(a, return_counts=True)
    print(u, indeces)


def binary_test():
    a, b = 13, 17;
    print(bin(a), bin(b))
    print('Bitwise AND of 13 and 17:')
    print(np.bitwise_and(13, 17))

    print(np.invert(np.array([13], dtype=np.uint8)))
    print('Binary representation of 13:')
    print(np.binary_repr(13, width=8))
    print(np.binary_repr(242, width=8))


def string():
    # string functions
    print(np.char.add(['hello'], ['xyz']))
    print(np.char.add(['hello', 'hi'], ['abc', 'xyz']))

    print(np.char.multiply('hello', 3), end=' ')

    print(np.char.center('hello', 20, fillchar='*'))
    print(np.char.capitalize('hello world'))
    print( np.char.title('hello how are you of course?'))
    print(np.char.split ('TutorialsPoint,Hyderabad,Telangana', sep = ','))

    print(np.char.strip('ashok arora', 'a'))
    print(np.char.join(':', 'dmy'))
    print(np.char.join([':', '-'], ['dmy', 'ymd']))

    print(np.char.replace('he is a good boy', 'is', 'was'))

    a = np.char.encode('hello', 'utf8')
    print(a)
    print(np.char.decode(a, 'utf8'))

def str():
    a = np.array([1.0,5.55, 123, 0.567, 25.532])
    print(np.around(a))
    print(np.around(a, decimals=1))
    print(np.around(a, decimals=-1))

    a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
    print(a)
    print(np.floor(a))
    print(np.ceil(a))

    a = np.array([[3,7,5],[8,4,3],[2,4,9]])
    print(a)

    print(np.amin(a, 1))
    print(np.amin(a, 0))

    a = np.array([[3,7,5],[8,4,3],[2,4,9]])
    print(a)
    print(np.ptp(a))
    print(np.ptp(a, axis=1))
    print(np.ptp(a, axis=0))

    # Order parameter in sort function
    dt = np.dtype([('name', 'S10'),('age', int)])
    a = np.array([("raju",21),("anil",25),("ravi", 17), ("amar",27)], dtype = dt)

    print('Our array is:')
    print(a)

    print('Order by name:')
    print(np.sort(a, order = 'name'))


def copy_and_view():
    a = np.arange(6)
    print(a)
    print(id(a))
    b = a;
    b.shape = (3, 2)
    print(b)
    print(a)

    # View or Shallow Copy
    a = np.arange(6).reshape(3, 2);
    print(a);
    b = a.view();
    print(b)
    print(id(a))
    print(id(b))

    # Slice of an array creates a view.
    a = np.array([[10, 10], [2, 3], [4, 5]]);
    print(a)
    s = a[:, :1]
    print(s)
    print(id(a))
    print(id(s))

    a = np.array([[10, 10], [2, 3], [4, 5]])
    b = a.copy();
    print(id(a))
    print(id(b))

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[11, 12], [13, 14]])
    print('dot')
    print(np.dot(a, b))
    print('matmul')
    print(np.matmul(a, b))

    a = [[1,0],[0,1]]
    b = [1,2]
    print(np.dot(a, b))
    print(np.dot(b, a))
    print(np.matmul(a, b))
    print(np.matmul(b, a))

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[11, 12], [13, 14]])
    print(np.vdot(a, b))














