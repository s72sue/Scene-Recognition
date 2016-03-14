# Script to convert a binary proto file to an npy file
import caffe
import numpy as np
import sys
from caffe.proto import caffe_pb2

if len(sys.argv) != 3:
    print "Usage: python convert_protomean.py mean.binaryproto out.npy"
    sys.exit()


data = open( sys.argv[1], 'rb').read()
blob = caffe_pb2.BlobProto()
blob.ParseFromString(data)
arr = caffe.io.blobproto_to_array(blob)
out = arr[0]
np.save(sys.argv[2], out)
print "shape of output: ", out.shape

