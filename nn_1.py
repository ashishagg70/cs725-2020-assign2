import sys
import os
import numpy as np
import pandas as pd
import time
np.random.seed(42)

NUM_FEATS = 90

class Net(object):
	def __init__(self, num_layers, num_units):
		self.num_out=1
		self.num_layers=num_layers
		self.num_units=num_units
		self.Wh = [np.random.uniform(-1, 1, (NUM_FEATS, num_units))]
		self.Bh=[np.random.uniform(-1, 1, (1,num_units))]
		for i in range(num_layers-1):
			self.Wh.append(np.random.uniform(-1, 1, (num_units, num_units)))
			self.Bh.append(np.random.uniform(-1, 1, (1,num_units)))
		self.Wh.append(np.random.uniform(-1, 1, (num_units, self.num_out)))
		self.Bh.append(np.random.uniform(-1, 1, (1,self.num_out)))
		self.net= {}
		self.out= {}
		self.del_W={}
		self.del_b= {}



	def __call__(self, X):
		self.net[0]=np.array(X.dot(self.Wh[0]) + self.Bh[0])
		self.out[0]=np.array(self.activation(self.net[0]))
		for i in range(1,self.num_layers+1):
			self.net[i]=np.array(self.out[i - 1].dot(self.Wh[i]) + self.Bh[i])
			self.out[i]=np.array(self.activation(self.net[i]))
		return self.net[self.num_layers]


	def backward(self, X, y, lamda):
		self.__call__(X)
		m=X.shape[0]
		i=self.num_layers
		dLdout = -2*(y-self.net[i])
		dLdW = self.out[i-1].T.dot(dLdout)/m
		dLdb = np.sum(dLdout, axis=0, keepdims=True) / m
		self.del_W[i]=dLdW+2*lamda*self.Wh[i]
		self.del_b[i]=dLdb+2*lamda*self.Bh[i]
		nextSumdLdV_dVdU=(dLdout.dot(self.Wh[i].T)) #mXN
		i-=1
		while(i>=0):
			dLdnet=np.maximum(0,nextSumdLdV_dVdU)
			if(i==0):
				dLdW = X.T.dot(dLdnet) / m
			else:
				dLdW = self.out[i - 1].T.dot(dLdnet) / m
			dLdb = np.sum(dLdnet, axis=0, keepdims=True) / m
			self.del_W[i] = dLdW + 2*lamda * self.Wh[i]
			self.del_b[i] = dLdb + 2*lamda * self.Bh[i]
			nextSumdLdV_dVdU=dLdnet.dot(self.Wh[i].T)
			i-=1
		return (self.del_W, self.del_b)

	def activation(self, Z):
		return np.maximum(0,Z)


class Optimizer(object):
	def __init__(self, learning_rate):
		self.learning_rate=learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		length=len(weights)
		i=0
		weight=[]
		bias=[]
		while(i<length):
			weight.append(weights[i] - self.learning_rate * delta_weights[i])
			bias.append(biases[i] - self.learning_rate * delta_biases[i])
			i+=1
		return (weight, bias)


def loss_mse(y, y_hat):
	return ((y-y_hat).T.dot(y-y_hat)).squeeze()/y.shape[0]

def loss_regularization(weights, biases):
	regulariser=0
	for w, b in zip(weights, biases):
		regulariser+=(w**2).sum()+(b**2).sum()
	return regulariser


def loss_fn(y, y_hat, weights, biases, lamda):
	return loss_mse(y, y_hat)+lamda*loss_regularization(weights, biases)

def rmse(y, y_hat):
	return np.sqrt(loss_mse(y, y_hat))

def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	begin = time.time()
	for t in range(max_epochs):
		sample_train_input, sample_train_targets = sample_batch(train_input, train_target, batch_size)
		i=0
		length=len(sample_train_targets)
		while(i<length):
			list_del_W, list_del_b = net.backward(np.array(sample_train_input[i]), np.array(sample_train_targets[i]), lamda)
			net.Wh, net.Bh=optimizer.step(net.Wh, net.Bh, list_del_W, list_del_b)

			i+=1
		y_hat=net(train_input)
		y_hat_dev = net(dev_input)
		print("dev loss: {:.5f}".format(rmse(dev_target, y_hat_dev)), "train loss: {:.5f}".format(rmse(train_target, y_hat)))
	print("time: ", time.time() - begin)
def get_test_data_predictions(net, inputs):
	y_hat = net(inputs)
	prediction_df = pd.DataFrame(data=y_hat, columns=["Predicted"])
	prediction_df.insert(0, "Id", np.arange(1, len(y_hat)+1, 1.0), True)
	prediction_df["Id"] = prediction_df["Id"].astype(str)
	prediction_df.to_csv("part2.csv",index=False)
	return y_hat

def read_data():
	df_train = pd.read_csv('dataset/train.csv', sep=r'\s*,\s*', engine='python')
	train_target=df_train['label'].to_numpy()
	train_target=train_target.reshape((len(train_target),1))
	df_train.drop('label', axis='columns', inplace=True)
	train_input = df_train.to_numpy()

	df_dev = pd.read_csv('dataset/dev.csv', sep=r'\s*,\s*', engine='python')
	dev_target = df_dev['label'].to_numpy()
	dev_target = dev_target.reshape((len(dev_target), 1))
	df_dev.drop('label', axis='columns', inplace=True)
	dev_input=df_dev.to_numpy()

	df_test = pd.read_csv('dataset/test.csv', sep=r'\s*,\s*', engine='python')
	test_input=df_test.to_numpy()
	return train_input, train_target, dev_input, dev_target, test_input

def sample_batch(train_input, train_target, batch_size):
	train_len=train_input.shape[0]
	m=batch_size
	no_of_split = train_len//m
	input_list=[]
	target_list=[]
	for i in range(no_of_split):
		start=i*batch_size
		end=start+batch_size
		input_list.append(train_input[start:end])
		target_list.append(train_target[start:end])
	if(train_len%m!=0):
		input_list.append(train_input[no_of_split*m:])
		target_list.append(train_target[no_of_split*m:])
	return (input_list, target_list)


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128
	train_input, train_target, dev_input, dev_target, test_input = read_data()
	print("data loaded...")

	learning_rate = .01
	num_layers = 1
	num_units = 64
	lamda = 0

	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	#get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
