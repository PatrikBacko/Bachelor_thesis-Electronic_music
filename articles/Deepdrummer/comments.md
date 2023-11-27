# Introduction: 
a program that can be used to create drum loops, with user preferencies.
Uses no dataset of drum loops, only a dataset of drum one-shots

## Three main parts:
- *generator* 
  - generates drum loops pattern with 4 drum one-shots and 16 time-stamps
  - MCMC with Metropolis-Hastings algorithm, monte carlo method with given probability distribution (probably generated by the neural network "critic") initial distribution is uniform 

- *critic* 
  - neural network that evaluates the generated drum loop (sound wave)
  - output is a probability of user liking the drum loop ([0,1] interval)
  - binary label (like/dislike)
  - input is a *MFCC* features of sound wave made by the pattern and one-shots

- *feature preprocessing*
  - input - loop pattern and with drum one-shots
  - output is a sound wave represented by *MFCC* features
    - !TODO: READ about MFCC features!

## Using:
- user gets a generated drum loop (by the generator) and evaluates it (like/dislike)
- the loop is evaluated by the critic and the critic is trained with label given by the user
- repeat until user is satisfied with the generated drum loop

## TODO (more reading):
- MCMC with Metropolis-Hastings algorithm
- **MFCC features** !!
- surf through the articles in the related work once more !!

- https://cs.wikipedia.org/wiki/Markov_chain_Monte_Carlo DONE
- https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm DONE
- https://dl.acm.org/doi/abs/10.1145/2557500.2557544
  - active learning, user feedback !!
- https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf

## interesting parts:
- **MFCC features** !! *** (possible feature extraction for my work)
- neural network critic trained by user feedback 
  - (maybe discus if something similar could be done for my work, some kind of fine-tuning of the model by user feedback) ***
- survey with participants to evaluate the system (could do something similar for my work, with artist friends, but simplier) ***