# Deep Learning Coronavirus Cure

This is an original submission to the Coronavirus Deep Learning Competition hosted by [Sage Health](https://www.sage-health.org/). The goal is to create a novel small molecule which can bind with the coronavirus, using deep learning techniques for molecule generation and [PyRx](https://pyrx.sourceforge.io/home) to evaluate binding affinities.

Binding scores of leading existing drugs (HIV inhibitors) are around -10 to -11 (the more negative the score the better), and around -13 for the drug Remdesivir which recently entered clinical testing.

By combining a generative RNN model with techniques and principles from transfer learning and genetic algorithms, I was able to create several small molecule candidates which achieved binding scores approaching -18.

## Acknowledgements

This repo began as a clone of [Topazape's](https://github.com/topazape/LSTM_Chem) repo which implements the paper [Generative Recurrent Networks for De Novo Drug Design](https://doi.org/10.1002/minf.201700111).

I also used [PyRx](https://pyrx.sourceforge.io/home) for all the binding affinity evaluation.

My thanks to Topazape, PyRx, and Sage Health for making this exciting work possible!

## Requirements

Requirements are identical to the original Topazape repo.

## Changes to Original Files

- Added a timeout feature in the cleanup_smiles.py script as in some cases of invalid smiles script would just hang
- Updated the max length of smiles used for training from 74 characters to 128 characters (this made the generative model much more robust)
- Modified notebook to use GPU over CPU
- Various parameter changes to RNN network (no architecture changes though)

## Approach

### Prepare Data - 'Data Staging.ipynb'

The original network only trained on ~450k unique smiles. My first goal was to train a network from scratch that would be highly adept at generating robust, realistic molecules.

I combined data sets from two sources: i) [Moses data set](https://github.com/molecularsets/moses) and ii) [ChEMBL data set](https://www.ebi.ac.uk/chembl/). Together these two data sets represented about 4 millions smiles.

After cleaning the smiles using the cleanup_smiles.py script and only retaining smiles between 34 to 128 characters in length, './datasets/all_smiles_clean.smi' contains the final list of ~2.5 million smiles on which the initial network was trained.

### Train Initial Network - 'Initial Network.ipynb'

I made no changes to the model architecture to train the baseline model, but did change several parameters to improve performance such as switching to GPU and increasing batch size. It took ~3-4 days to train 25 epochs for the ~2.5million SMILES on a single GTX 1060.

I had originally planned to train for up to 40 epochs, but by 22 epochs the validation loss flattened off and subsequently started to rise, so I used a final model saved after 22 epochs. It's possible this was a local minimum, but the performance of the network in generating realitic molecules was quite high so I did not continue to train it due to limited time.

### Generating Initial Universe of SMILES - 'Initial Network.ipynb'

After completing training, I used the new network to generate 10,000 smiles. I would have liked to generate more to start with a wider set of molecules to evaluate before narrowing in on molecules that bind well, but time was a constaint as the generation process takes several hours.

I evaluated relative performance of the original repo network vs my new network along two metrics from the original repo, and added a third metric of my own creation:

- Validity: of the total number of generated smiles, percentage that are actually valid smiles for molecules
- Uniqueness: of the total valid number of generated smiles, percentage that are not duplicates
- Originality: of the total number of valid generated smiles, percentage that are brand new creations that do not appear in the training data

Original LSTM_Chem network benchmarks:

- Validity: 62.3%
- Uniqueness: 99.8%

My newly trained network metrics:

- Validity: 97.0%
- Uniqueness: 99.8%
- Originality: 89.0%

Originally generated 10,000 smiles are saved to './generations/gen0_smiles.smi'

### Finding Top Candidates from Initial Universe of SMILES - 'Evaluation and Refinement.ipynb'

Having generated ~10k new valid molecules, my biggest constraint was time: evaluating each molecule's binding affinity with the coronavirus protease via PyRx is a lengthy process with ~1.5 molecules evaluating per minute. Running an analysis of 10k molecules was therefore not possible as that would take over 100 hours.

In order to minimize time the function initialize_generation_from_mols() randomly picks 30 molecules, then iterates through the rest of the list calculating that molecules Tenimoto Similarity scores to the molecules so far added to the list, and only 'accepts' the molecule if the maximum similarity score is less than a certain threshold. This ensures that even a smaller sample will feature a diverse set of molecules.

I was then able to save smiles to a pandas dataframe (csv), and also convert smiles to molecules and write these molecules to an SDF file which could be manually imported into PyRx for analysis. PyRx then outputs a csv of molecules and their binding scores after analysis. In order to relate smiles in my pandas/csv to molecules as SDF in PyRx, I used Chem.PropertyMol.PropertyMol(mol).setProp() to set the 'Title' property to a unique identifier of four letters and a generation number.

### Evaluating Molecule <> Coronvirus Fit - PyRx

I used PyRx to evalaute molecule binding scores. I recommend this tutorial to get started:
[PyRX ligand docking tutorial](https://www.youtube.com/watch?v=2t12UlI6vuw)

### Transfer Learning & Genetic Algorithm - 'Evaluation and Refinement.ipynb'

After evaluating about 1500 gen0 smiles in PyRx (an overnight task), I had a variety of scores for a diverse set of molecules. I then employed techniques and principles of genetic algorithms and transfer learning to take the original network's knowledge of creating realistic molecules and transfer it to the domain of making molecules specifically adept at binding with the coronavirus.

Each generation I ran the following steps:

1. Ranked all molecules so far tested across all generations by binding scores, and picked the top X smiles with the best scores (I used 35).
2. Calculate the similarity of each remaining molecule with the set of molecules from Step 1, and calculate an adjusted score that boosts molecules that are very different than the top ranking molecules and have good but not great scores (ie they may work via a different mechanism so keep exploring that mechanism). Take the top X smiles ranked by this similarity adjusted score (I used 5).
3. In basic research, I learned that one of the most critical defining charactersitics of small molecules is that they weigh less than 900 daltons. I noticed that larger molecules (1000-1050 range) seemed to be getting high binding affinfity scores, so in order to both learn from what made those large molecules good, but also promote smaller molecules, I computed a weight adjusted score that boosts light weight molecules with good but not great scores. I then ranked by this adusted score and took the top X molecules (I used 5).
4. The above steps yielded a list of 45 molecules considered 'good fits' across three metrics of fit: i) overall score, ii) similarity adjusted score (ensuring diverse molecules are included), iii) weight adjusted score (ensuring especially small molecules are included). As a way to promote random 'mutations' (inspired by a genetic algomith approach) I used the _baseline_ model to generate a random sample of molecules in each generation. (I took 5)
5. Now we have 50 total 'target' smiles (ie the 'parents' for our genetic algorithm). I then cumulatively the previous generation's network on these 50 target smiles. I applied a rule-of-thumb to train the network enough to cut its loss in half from the frist epoch to the last epoch each time. By trial-and-error, I found this to typically be ~5 epochs, so that's what I used.
6. After training a new model on the generation's well-fit "parents", I used it to generate the next generation of ideally better-fit "children". As a rule of thumb I would generate 500 smiles each generation, which after removing duplicates, invalids, etc usually led to a few hundred children to evaluate per generation.
7. Save the new generation to the tracking pandas/csv and to molecule SDF, then feed SDF into PyRx and evaluate.

I repeated the above steps across 10 generations, each time using the best fit + 'mutation' training set from the prior generation to teach the network to create molecules better and better at binding.

## Final Results - 'Final Results.ipynb'

Here I took the top candidates I'd found (score <-15 and weight <900 daltons) and reran them in PyRx, splitting out columns for each molecules best binding score, and its average binding score (its average across the molecule's modes in PyRx), and each molecules similarity to existing HIV inhibitor drugs and the drug Remdesivir which is currently in clinical testing.

As you can see, the model generated significantly better binding scores than existing drugs.

![final results PyRx screenshot](https://github.com/mattroconnor/deep_learning_coronavirus_cure/blob/master/generations/results/final_results.JPG "PyRx Final Results")

See './generations/master_results_table_final.csv' for full table of final results.

## Next Steps

- Have a domain expert analyze top findings for fit and/or find the molecules in the universe of existing approved drugs which are most similar to top findings and evaluate them for fit
- According to [this paper](https://arxiv.org/pdf/1703.07076.pdf) the baseline model may be further improved by training on a universe of enumerated smiles, not just canonical smiles
- Code needs refactoring

## About Me (Matt O'Connor)

I am a data science and strategy coach who guides businesses through the delivery of real world projects to 10x their internal data & AI capabilities while delivering immediate value.

I coach:

- Executives: on data & AI strategy, leadership, organizational transformation, and culture
- Business Teams: on data team structures, solution design, and project management
- Tech Teams: on data/ML/AI architecture, deployment, infrastructure, and tools

My experience includes having served as lead data scientist of algorithmic trading for Ray Dalio’s \$160bn hedge fund Bridgewater Associates, raising VC funding at 7-figure valuations, and coaching some of the world's largest multinationals in the supply chain, retail, banking & insurance, and real estate industries on data transformations.

It my sincere hope my work helps us collectively advance the fight against coronavirus. If you reuse or republish my findings or work, please attribute me: "Matt O'Connor, Founder of https://Reboot.ai"

My github is almost exclusively contractually private projects, but to get in touch please visit [Reboot AI](https://www.reboot.ai/) or connect with me on [LinkedIn](https://www.linkedin.com/in/mattroconnor/).

## Other Helpful Links (arbitary order)

https://github.com/EBjerrum/SMILES-enumeration

https://downloads.ccdc.cam.ac.uk/documentation/API/descriptive_docs/docking.html#rescoring

https://life.bsc.es/pid/pydock/index.html

https://github.com/brianjimenez/pydock_tutorial

https://www.rdkit.org/docs/GettingStartedInPython.html

https://www.rcsb.org/pdb/explore/sequenceCluster.do?structureId=6LU7&entity=1&seqid=95

https://docs.google.com/document/d/1Ni77kjCfAvSFSXAkZsboDQ80UUu2YGJ232jS499TR5Y/edit

https://github.com/sarisabban/Notes/blob/master/AutoDock.py

http://autodock.scripps.edu/faqs-help/how-to/how-to-setup-adt-scripts

https://omictools.com/binana-tool

https://sourceforge.net/p/rdkit/mailman/message/36335970/
