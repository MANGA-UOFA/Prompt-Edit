from config import get_args
opt=get_args()
dst=opt.dst
if dst=='yelp' or dst=='amazon':
    stopwords='#$%&()*+--)–/:;<=>@[\\]^_`{|}~—•…�™—'+'0123456789'

DATASET_DICT ={
    'yelp': {
        'from': ['positive', 'negative'],
        'from_to': {
            'positive': 'negative',
            'negative': 'positive',
            },
        'examples': [
            ('negative', 'this place is awful!'),
            ('positive', 'this place is amazing!', 'negative'),
            ('negative', 'i hated their black tea and hated hot chocolate selections!'),
            ('positive', 'i loved their black tea and loved hot chocolate selections!'),
        ],
    },
    'amazon': {
        'from': ['positive', 'negative'],
        'from_to': {
            'positive': 'negative',
            'negative': 'positive',
            },
        'examples' : [
            ('negative', 'this place is awful!'),
            ('positive', 'this place is amazing!', 'negative'),
            ('negative', 'i hated their black tea and hated hot chocolate selections!'),
            ('positive', 'i loved their black tea and loved hot chocolate selections!'),
        ],
    },
}


def write_sentence(dataset, delim_left, delim_right, orig_text, rewritten_text=None):

    if dataset=='yelp' or dataset=='amazon':
        style_word='sentiment'

    sentence =f'The {style_word} of the text {delim_left}{orig_text}{delim_right} is: '

    if rewritten_text is not None:
        sentence = f'{sentence} {rewritten_text}'
    return sentence

# EOSequence token
FS_EOS_TOKEN = '\n###\n'

# Create exemplars (for the few-shot setting)
def create_exemplars(dataset, num_examples, delim_left, delim_right):
    prefix = ''
    exemples=DATASET_DICT[dataset]['examples'][:num_examples]
    for exemple in exemples:
        # ('negative', 'this place is awful!', 'positive', 'this place is amazing!'),
        orig_style, orig_text, _, _ = exemple
        add_text = write_sentence(dataset, delim_left, delim_right, orig_text, orig_style)
        prefix += f'{add_text}{FS_EOS_TOKEN}'
    return prefix



