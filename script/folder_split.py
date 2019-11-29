import split_folders


"""
 To install split_folders package--
 ! pip install split-folders

"""
# Split with a ratio.
# To only split into training ,testing and validation set, set a tuple to `ratio`, i.e, `(.8, .1,.1)`.

split_folders.ratio('input_folder', output="output", seed=1337, ratio=(.8, .1, .1)) # default values
