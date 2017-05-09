import sys
import tensorflow as tf

from PIL import Image, ImageFilter


def _prepare_image(image_path):
    """
    :param image_path: path to image
    :return: pixels values
    """
    image = Image.open(image_path)
    image = image.convert('L')
    new_image = _get_normalized_image(image)
    pixels = list(new_image.getdata())
    normalized_pixels = [(255 - x) * 1.0 / 255.0 for x in pixels]
    return normalized_pixels


def _get_normalized_image(image):
    """
    :param image
    :return: normalized image
    """
    width = float(image.size[0])
    height = float(image.size[1])
    new_image = Image.new('L', (28, 28), 255)
    if width > height:
        new_height = int(round((20.0 / width * height), 0))
        if new_height == 0:
            new_height = 1
        tmp_image = image.resize((20, new_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        width_top = int(round(((28 - new_height) / 2), 0))
        new_image.paste(tmp_image, (4, width_top))
    else:
        new_width = int(round((20.0 / height * width), 0))
        if new_width == 0:
            new_width = 1
        tmp_image = image.resize((new_width, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        width_left = int(round(((28 - new_width) / 2), 0))
        new_image.paste(tmp_image, (width_left, 4))
    return new_image


def _predict_integer(image_pixels):
    """
    :param image_pixels
    :return: prediction result
    """

    # define the model (same as in create_model.py)

    x = tf.placeholder(tf.float32, shape=[None, 784])

    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv2d(x_arg, w_arg):
        return tf.nn.conv2d(x_arg, w_arg, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(x_arg):
        return tf.nn.max_pool(x_arg, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = _weight_variable([5, 5, 1, 32])
    b_conv1 = _bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = _max_pool_2x2(h_conv1)

    W_conv2 = _weight_variable([5, 5, 32, 64])
    b_conv2 = _bias_variable([64])

    h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = _max_pool_2x2(h_conv2)

    W_fc1 = _weight_variable([7 * 7 * 64, 1024])
    b_fc1 = _bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = _weight_variable([1024, 10])
    b_fc2 = _bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'saved_models/model.ckpt')
    prediction = tf.argmax(y_conv, 1)
    return (prediction.eval(feed_dict={x: [image_pixels], keep_prob: 1.0}, session=sess))[0]


def _main(argv):
    prepared_image = _prepare_image(argv)
    prediction_result = _predict_integer(prepared_image)
    print('---prediction result: ', prediction_result)


if __name__ == "__main__":
    _main(sys.argv[1])
