import glob
import errno
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

data_path = 'eventdata/*.txt'
def load_data():
	x=[]
	y=[]
	label=[]
	files = glob.glob(data_path)
	for name in files:
		try:
			with open(name) as f:
				label.append(name[name.find('_')+1:name.find('.txt')])
				tmp_x=[]
				max=[0,0,0,0,0,0,0,0]
				for line in f:
					tmp_l=map(float,line.replace('\r\n', '').split('\t'))
					for e in range(0,8):
						if tmp_l[e+1]>max[e]:
							max[e]=tmp_l[e+1]
					if tmp_l > 0:
						tmp_x.append(tmp_l[1:])
				x.append(np.array(tmp_x)/np.array(max))
				y.append(int(f.name[f.name.find('/')+1:][:f.name[f.name.find('/')+1:].find('_')]))
		except IOError as exc:
			if exc.errno != errno.EISDIR:
				raise

	ycat=np_utils.to_categorical(y)
	ycat=np.delete(ycat,0,1)

	x=np.array(x)
	y=np.array(y)
	train_x_id=[]
	test_x_id=[]
	valid_x_id=[]
	ulabels=list(set(label))
	count=0
	for vid in range(0,len(ulabels)):
		count=count+1
		for ind in range(0,len(y)):
			if label[ind]==ulabels[vid]:
				if count%2==0 and y[ind]%2==0:
					if y[ind]%6==0:
						valid_x_id.append(ind)
					elif y[ind]%2==0:
						test_x_id.append(ind)
				elif count%2!=0 and y[ind]%2!=0:
					train_x_id.append(ind)

	x_red=[]
	for i in range(0,len(x)):
		x_instance=[]
		for j in range(0, len(x[i])):
			x_instance.append(x[i][j][[0,1,2,4,5,6]])
		x_red.append(x_instance)

	x_exp_pad=[]
	for itr in range(0,len(x)):
		x_exp_pad.append(len(x[itr]))

	x_exp_pad=np.array(x_exp_pad)
	max_len=np.max(x_exp_pad)
	x_exp_pad=max_len-x_exp_pad
	x_mask=np.zeros((len(x),max_len-((3-1)*16),128))
	for itr in range(0,len(x)):
		if x_exp_pad[itr] >0:
			x_mask[itr,-1*x_exp_pad[itr]:,:]=-999999

	x_mask=np.array(x_mask)
	x_red=pad_sequences(x_red, padding='post',dtype=float)
	
	return x_red,y,train_x_id,test_x_id,valid_x_id,x_mask,x_exp_pad