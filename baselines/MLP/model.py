import torch
import torch.nn as nn


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)
		m.bias.data.fill_(0.01)

def build_metaclassifier(input_size, lr):
	MLP = nn.Sequential(
		nn.Linear(input_size, input_size//2),
		nn.Linear(input_size//2, input_size//4),
		nn.Dropout(),
		nn.Linear(input_size//4, 1),
		nn.Sigmoid()
	)
	
	MLP.apply(weight_init)
	MLP.train()
	MLP.cuda()

	loss_fn = nn.BCELoss()

	optimizer = torch.optim.SGD(MLP.parameters(), lr=lr, momentum=0.9)

	return MLP, loss_fn, optimizer