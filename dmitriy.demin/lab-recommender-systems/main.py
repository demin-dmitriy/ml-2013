import sys
from math import sqrt

class TrainData:
	def __init__(self, input):
		# Parse input
		lines = input.split('\n')
		self.max_rating, self.nusers, self.nitems, ntrainrates, ntestrates = \
			map(int, lines[0].split())
		self.train_rates = [[None] * self.nusers for j in range(self.nitems)]
		for user, item, rate in (map(int, line.split()) for line in lines[1:ntrainrates+1]):
			self.train_rates[item][user] = rate
		self.test_rates = [(int(user), int(item)) for user, item in \
			(line.split() for line in lines[ntrainrates+1:ntrainrates+1+ntestrates])]

# Predicts rates using weighted average of neighbors' ratings
def answer(tests, max_rating, train_rates, item_correlation):
	# n = 0
	for user, item in tests:
		# n += 1
		# print("Debug: ", n, user, item)
		if train_rates[item][user] != None:
			yield train_rates[item][user]
			continue
		sum_rate = 0.
		sum_weight = 0.
		for i, item_rates in enumerate(train_rates):
			if item_rates[user] != None:
				sum_rate += item_correlation.of(i, item) * item_rates[user]
				sum_weight += item_correlation.of(i, item)
		if sum_weight == 0.: # User hasn't rated any item
			yield max_rating / 2
		else:
			yield sum_rate / sum_weight

def dot_product(a, b):
	return sum(x * y for x, y in filter(lambda p: None not in p, zip(a, b)))

class ItemCorrelation:
	def __init__(self, train_data):
		self.train_data = train_data
		# list of pairs (rates' deviation, rates' mean)
		self.item_params = [sqrt(sum(x**2 for x in filter(None, item_data))) for item_data in train_data.train_rates]
		# Estimates the simmilarity between items using Pearson's coefficient
		self.item_correlation = [[None] * train_data.nitems for i in range(train_data.nitems)]

		
	def of(self, i, j):
		if self.item_correlation[i][j] == None:
			i_sqr_mean = self.item_params[i]
			j_sqr_mean = self.item_params[j]
			if i_sqr_mean == 0 or j_sqr_mean == 0:
				ij_cor = 0
			else:
				ij_cor = dot_product(self.train_data.train_rates[i], self.train_data.train_rates[j]) / (i_sqr_mean * j_sqr_mean)
			self.item_correlation[i][j] = ij_cor
			self.item_correlation[j][i] = ij_cor
		return self.item_correlation[i][j]

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: {} <test_data_file>".format(sys.argv[0]))
		sys.exit(1)

	train_data = TrainData(open(sys.argv[1]).read())
	item_correlation = ItemCorrelation(train_data)
	for rate in answer(train_data.test_rates, train_data.max_rating, train_data.train_rates, item_correlation):
		print(str(rate))
