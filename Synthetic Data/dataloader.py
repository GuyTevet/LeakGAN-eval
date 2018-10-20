import numpy as np
import os


class Gen_Data_loader():
    def __init__(self, batch_size,length):
        self.batch_size = batch_size
        self.token_stream = []
        self.length = length
    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.length:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

class Gen_Data_loader_text8(Gen_Data_loader):

    def __init__(self, batch_size, charmap, inv_charmap,seq_len=20):
        super(Gen_Data_loader_text8, self).__init__(batch_size,seq_len)
        self.charmap, self.inv_charmap = charmap, inv_charmap
        self.seq_len = seq_len

    def create_batches(self, data_file, limit_num_samples=None):

        cache_file = "%s_seqlen%0d.npy"%(data_file,self.seq_len)

        if os.path.exists(cache_file):
            self.token_stream = np.load(cache_file)
        else:
            self.token_stream = []

            with open(data_file, 'r') as f:
                line = f.read(self.seq_len)
                while len(line) == self.seq_len:
                    tokens = [int(self.charmap[char]) for char in line]
                    assert len(tokens) == self.seq_len
                    self.token_stream.append(tokens)

                    line = f.read(self.seq_len)

            self.token_stream = np.array(self.token_stream)
            np.save(cache_file.replace('.npy',''),self.token_stream)

        assert self.token_stream.shape[1] == self.seq_len

        if limit_num_samples is not None:
            # choose only limit_num_samples from them
            self.token_stream = np.array(self.token_stream)
            permut = np.random.permutation(self.token_stream.shape[0])[:limit_num_samples]
            self.token_stream = self.token_stream[permut]
        else:
            self.token_stream = np.array(self.token_stream)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

        print("done create_batches - [num_batch=%0d]"%self.num_batch)

    def reset_pointer(self):
        self.shuffle()
        self.pointer = 0

    def shuffle(self):
        print("GEN shuffling data...")
        permut = np.random.permutation(self.token_stream.shape[0])
        self.token_stream = self.token_stream[permut]


class Dis_dataloader(object):
    def __init__(self, batch_size,length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.length = length
    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader_text8(Dis_dataloader):

    def __init__(self, batch_size, charmap, inv_charmap,seq_len=20):
        super(Dis_dataloader_text8, self).__init__(batch_size,seq_len)
        self.charmap, self.inv_charmap = charmap, inv_charmap
        self.seq_len = seq_len

    def load_train_data(self, positive_file, negative_file):

        # #LOAD NEGATIVE
        # positive file is constant while negative file is changed during train!
        # if os.path.exists(negative_file + '.npy'):
        #     negative_examples = np.load(negative_file + '.npy')
        # else:
        negative_examples = []

        # remove \n
        with open(negative_file, 'r') as f:
            all_negative = f.read()
        with open(negative_file, 'w') as f:
            f.write(all_negative.replace('\n',''))

        with open(negative_file, 'r') as f:
            line = f.read(self.seq_len)
            while len(line) == self.seq_len:
                tokens = [int(self.charmap[char]) for char in line]
                assert len(tokens) == self.seq_len
                negative_examples.append(tokens)

                line = f.read(self.seq_len)

        # np.save(negative_file,np.array(negative_examples))
        negative_examples = np.array(negative_examples)
        num_positive_samples = negative_examples.shape[0]


        #LOAD POSITIVE

        cache_positive = "%s_seqlen%0d.npy"%(positive_file,self.seq_len)

        if os.path.exists(cache_positive):
            positive_examples = np.load(cache_positive)
        else:
            positive_examples = []

            with open(positive_file, 'r') as f:
                line = f.read(self.seq_len)
                while len(line) == self.seq_len:
                    tokens = [int(self.charmap[char]) for char in line]
                    assert len(tokens) == self.seq_len
                    positive_examples.append(tokens)

                    line = f.read(self.seq_len)

            positive_examples = np.array(positive_examples)
            np.save(cache_positive.replace('.npy',''),positive_examples)

        assert positive_examples.shape[1] == self.seq_len

        #choose only num_positive_samples from them
        permut = np.random.permutation(positive_examples.shape[0])[:num_positive_samples]
        positive_examples = positive_examples[permut]

        # CONCAT
        negative_examples = np.array(negative_examples)
        positive_examples = np.array(positive_examples)
        assert negative_examples.shape == positive_examples.shape
        self.sentences = np.concatenate((positive_examples,negative_examples),axis=0)

        # Generate labels
        positive_labels = [[0, 1]] * positive_examples.shape[0]
        negative_labels = [[1, 0]] * negative_examples.shape[0]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # # Shuffle the data
        # print "DISC shuffling data..."
        # shuffle_indices = np.random.permutation(self.sentences.shape[0])
        # self.sentences = self.sentences[shuffle_indices]
        # self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

        print("done create_batches - [num_batch=%0d]"%self.num_batch)

    def reset_pointer(self):
        self.shuffle()
        self.pointer = 0

    def shuffle(self):
        print("DISC shuffling data...")
        shuffle_indices = np.random.permutation(self.sentences.shape[0])
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

