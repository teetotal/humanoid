import h5py

def h5_read(path):
    return h5py.File(path, 'r')

class h5_gen:
    def __init__(self, file_path, compression = None, mode = 'w'):
        self.groups = {}        
        self.f = h5py.File(file_path, mode) 
        self.compression = compression       

    def write(self, path, dataset_name, dataset):
        if (path in self.groups) == False:
            self.g = self.f
            arr = path.split(u'/')
            key = ''
            for name in arr:
                if name == '':
                    continue

                key += u'/' + name
                if (key in self.groups) == False:
                    self.g = self.g.create_group(name)
                    self.groups[key] = self.g
        
        self.groups[path].create_dataset(dataset_name, data=dataset, compression = self.compression)

        return self.groups[path]
    
    def read(self, path):
        return self.groups[path]
    def close(self):
        self.f.close()

'''
h5 = h5_gen('./test.h5')
data = [0, 1, 2, 3, 4, 5, 6, 7]
data1 = [0, 1, 2]
data2 = [3, 4, 5, 6, 7]

d = h5.write('/test/1', data)
d = h5.write('/test/1/1', data1)
d = h5.write('/test/1/2', data2)

d = h5.read('/test')
print(d)
d = h5.read('/test/1')
print(d)
d = h5.read('/test/1/1')
print(d)
d = h5.read('/test/1/2')
print(d)
d = h5.read('/test/1/3')
print(d)

파일을 w모드로 열기 때문에 프로세스가 끝나고 나서는 h5_read로 열어야 함
'''
