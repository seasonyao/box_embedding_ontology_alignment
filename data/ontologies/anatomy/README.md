# File Information

## **.pickle** files

- (human/mouse/entities).pickle
- None of the **.pickle** files contain train/dev split data. It is strictly referential.
- **human.pickle** contains all human entities, and the parents_of and children_of each entity in the human graph.
- **mouse.pickle** contains all mouse entities, and the parents_of and children_of each entity in the mouse graph.
- **entities.pickle** contains every edge, this includes edges within individual ontologies, and also across ontologies (alignments). It also contains a number of useful dictionaries that can be used to identify entities. For example *idx2label* can be used to index into the labels of every entity.

## **.tsv** files

- There is a large set of **.tsv** files within this folder. At first glance they may be overwhelming, but this description should help break them down.
- Unary probabilities for each individual ontology are stored in **{onto}_unary.tsv**.
- **tr_{pos/neg}_{pct}.tsv** refers to all positive and negative edges in the training set, these edges include edges within ontologies, along with alignment edges. **pct** refers to the train/dev split.
- **tr_align_{pos/neg}_{pct}.tsv** contains positive and negative alignments that show up in the training data set. Used for evaluation of how well the training alignments have been learned. **pct** refers to the train/dev split.
- **dev_align_{pos/neg}_{pct}.tsv** contains positive and negative alignments that show up in the development data set. **pct** refers to the train/dev split.
- When observing performance within an individual ontology (ie, human or mouse), there are specific files to load that contain train or dev edges for that given ontology.
- **{human/mouse}_dev\_{pos/neg}.tsv** are datasets that contain either positive of negative edges **(pos/neg)**  for a single ontology **(human/mouse)**. For example, **human_dev_neg.tsv** contains all positive edges for the human ontology alone. As the **dev** denotes, these datasets are primarily used as a reference, not for training. These datasets show how well the model learns the individual structure of each ontology while it is learning the main goal, alignments.
- The folder **individual_analysis/** contains datasets that are not for the alignment task. These datasets are adjusted so that any individual ontology is 0 indexed. **NOTE:** if you are using this dataset, the information stored in the *.pickle* files will not match up on indeces, so they will need to be adjusted if you want to look at that information.
