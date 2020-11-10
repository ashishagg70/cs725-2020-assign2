import numpy as np
import pandas as pd
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
		return np.around(self.net[self.num_layers])


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
			dLdnet = np.array(nextSumdLdV_dVdU, copy=True)
			dLdnet[self.net[i] <= 0] = 0
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
	#begin = time.time()
	for t in range(max_epochs):
		shuffle_input_ind = np.random.choice(train_input.shape[0], train_input.shape[0], replace=False)
		shuffled_train_input, shuffled_train_target=train_input[shuffle_input_ind], train_target[shuffle_input_ind]
		for i in range(0, train_input.shape[0], batch_size):
			list_del_W, list_del_b = net.backward(shuffled_train_input[i:i + batch_size], shuffled_train_target[i:i + batch_size], lamda)
			net.Wh, net.Bh = optimizer.step(net.Wh, net.Bh, list_del_W, list_del_b)

	y_hat=net(train_input)
	y_hat_dev = net(dev_input)
	dev_loss=rmse(dev_target, y_hat_dev)
	print("dev loss: {:.5f}".format(dev_loss), "train loss: {:.5f}".format(rmse(train_target, y_hat)))
	#print("time: ", time.time() - begin)
def get_test_data_predictions(net, inputs):
	y_hat = net(inputs)
	prediction_df = pd.DataFrame(data=y_hat, columns=["Predicted"])
	prediction_df.insert(0, "Id", np.arange(1, len(y_hat)+1, 1.0), True)
	prediction_df["Id"] = prediction_df["Id"].astype(str)
	prediction_df.to_csv("part2.csv",index=False)
	return y_hat

def read_data():
	df_train = pd.read_csv('dataset/train.csv', sep=r'\s*,\s*', engine='python')
	train_target = df_train['label'].to_numpy()
	train_target = train_target.reshape((len(train_target), 1))
	df_train.drop('label', axis='columns', inplace=True)
	mean = df_train.mean(axis=0)
	std = df_train.std(axis=0)
	df_train = (df_train - mean) / (std)
	min = df_train.min(axis=0)
	max = df_train.max(axis=0)
	df_train = (df_train - min) / (max - min)

	# print(df_train)

	train_input = df_train.to_numpy()

	df_dev = pd.read_csv('dataset/dev.csv', sep=r'\s*,\s*', engine='python')
	dev_target = df_dev['label'].to_numpy()
	dev_target = dev_target.reshape((len(dev_target), 1))
	df_dev.drop('label', axis='columns', inplace=True)
	df_dev = (df_dev - mean) / (std)
	df_dev = (df_dev - min) / (max - min)
	dev_input = df_dev.to_numpy()

	df_test = pd.read_csv('dataset/test.csv', sep=r'\s*,\s*', engine='python')
	df_test = (df_test - mean) / (std)
	df_test = (df_test - min) / (max - min)
	test_input = df_test.to_numpy()

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 100
	batch_size = 128
	train_input, train_target, dev_input, dev_target, test_input = read_data()
	print("data loaded...")

	learning_rate = .001
	num_layers = 1
	num_units = 128
	lamda = 0

	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)

if __name__ == '__main__':
	main()
