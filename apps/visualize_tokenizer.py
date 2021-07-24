import tensorflow as tf

from transformer.text.tokenizer import TokenizerBert

if __name__ == '__main__':
    import os
    import sys
    myPath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, myPath + '/../../')
    from datasets.scripts.fra_eng import datasets_fra_eng
    _, _, full_dataset = datasets_fra_eng()
    fr_dataset = full_dataset.map(lambda fr, _: fr)
    fr_tokenizer = TokenizerBert(lower_case=False)
    fr_tokenizer.build_tokenizer(fr_dataset.take(3000))
    fr_text = [
        "Nous sommes enthousiasmés par l'avenir des modèles basés",
        "sur l'attention et prévoyons de les appliquer à d'autres",
        "tâches. Nous prévoyons d'étendre les Transformers aux problèmes",
        "impliquant des modalités d'entrée et de sortie différentes",
        "de celles du texte, et d'étudier les mécanismes d'attention locale",
        "restreinte pour gérer efficacement des entrées et sorties",
        "volumineuses comme les images, l'audio et les vidéos. Rendre la",
        "génération moins séquentielle est un de nos autres objectifs",
        "de recherche."
        ]
    fr_dataset = tf.data.Dataset.from_tensor_slices(fr_text)
    print('vocab', fr_tokenizer.vocab)
    print('len vocab:', len(fr_tokenizer.vocab))
    print('\ntokens:')
    for text in fr_dataset.batch(2):
        tokens = fr_tokenizer.tokenize(text)
        print(tokens)
    print('\ntext:')
    for text in fr_dataset.batch(2):
        tokens = fr_tokenizer.tokenize(text)
        text_tensor = fr_tokenizer.detokenize(tokens)
        tf.print(text_tensor)