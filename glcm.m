I = imread('/home/t3min4l/workspace/texture-classification/data/subdataset/canvas1/canvas1-a-p001.png')
glcm = graycomatrix(I, 'Numlevels', 256)
mat2np(glcm, './glcm.pkl', 'int8')

