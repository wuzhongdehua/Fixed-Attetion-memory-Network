import data_unit
from model import Fixed_MemNN
import tensorflow as tf
import sys

flags = tf.app.flags

flags.DEFINE_integer("edim", 100, "encoding dimension")
flags.DEFINE_integer("nhop", 3, "number of hops")
#flags.DEFINE_integer("batch_size", 1920, "batch size")
flags.DEFINE_integer("nanswer", 5, "number of answer sentences")
flags.DEFINE_integer("nstory", int(sys.argv[3]), "number of story sentences")
flags.DEFINE_float("init_lr", (1e-2)/2, "initial learning rate")
flags.DEFINE_float("init_std", 0.05, "weight initialization std")
flags.DEFINE_float("gamma", 0 , 'weight decay gamma value')

if sys.argv[1] == 'train':
    flags.DEFINE_bool("train", True, "Training mode on.")
    flags.DEFINE_integer("batch_size", 32, "batch size")
elif sys.argv[1] == 'test':
    flags.DEFINE_bool("train", False, "Training mode on.")
    flags.DEFINE_integer("batch_size", 1920, "batch size")

if sys.argv[2] == 'word2vec' :
    flags.DEFINE_string("embedding_method", 'word2vec', 'word2vec embedding')
    flags.DEFINE_integer("idim", 2500, "input word2vec vector dimension")
if sys.argv[2] == 'skip' :
    flags.DEFINE_string("embedding_method", 'skip', 'skipthoughts embedding')
    flags.DEFINE_integer("idim", 4800, "input skip-thought vector dimension")

FLAGS = flags.FLAGS

class example(object):
  story = ["Kevin went into the kitchen", "Kevin washed dishes", "Kevin went into the bedroom"]
  query = ["What did Kevin do in the kitchen?"]
  answer = ["He cooked his lunch", "He cleaned the floor", "He washed dishes", "He took a walk", "He took out a bowl"]
  target = 3

def main(_):
  with tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
    device_count={'GPU': 2})) as sess:
    model = Fixed_MemNN(FLAGS, sess)
    model.build_model()
    data_engine = data_unit.Dataset(FLAGS.nstory)

    if FLAGS.embedding_method == 'word2vec': data_engine.load_dataset('word2vec')
    elif FLAGS.embedding_method == 'skip' : data_engine.load_dataset('skip')
    num_iter = 10000000
    if FLAGS.train == True:
        maximum_acc = 0.0
        print '[*] Training start.'
        for i in xrange(num_iter):
            loss = model.train(data_engine.next_batch(model.batch_size))

            if i%100 == 0:
                acc = model.test(data_engine.next_batch(1920, 'val')) * 100
                maximum_acc = max([maximum_acc, acc])
                print '[**] Maximum Accuracy >> %.2lf%%' % maximum_acc
                print 'At training step %d, Training loss : %lf' % (i, loss)
            if i%10000 == 0:
                model.train(data_engine.next_batch(model.batch_size), True, i)

if __name__ == '__main__':
  tf.app.run()
