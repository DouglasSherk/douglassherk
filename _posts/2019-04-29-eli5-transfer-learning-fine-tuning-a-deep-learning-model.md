---
layout: post
title: "ELI5 \u2013 Transfer Learning/Fine-Tuning a Deep Learning Model"
date: 2019-04-29
categories:
  - Deep Learning
source_url: https://datastronomy.com/eli5-transfer-learning-fine-tuning-a-deep-learning-model/
---

<p>Imagine that you work at a factory, and that your boss has a week-long task for you to sort large screws that are continuously coming down a conveyor belt and to place them into one of twenty labeled boxes. The boxes have labels with the names of colors like “red”, “green”, &#8220;blue&#8221;, etc., and each screw has a single colored band on it that matches up with exactly one of the boxes. You&#8217;re now on the hook to solve the problem, but neither you nor any human is fast enough to keep up with the hundreds of screws coming down the conveyor belt every minute. </p>



<p>You, being the smart person you are, remember that you have a couple of baby nephews who are free for the summer, so you enlist their help. The three of you working together should be able to complete your task. There’s only one problem: the babies don’t yet know how to identify colors, and so they can’t sort the screws. You decide to first teach the infants common colors and their respective names.</p>



<p>You give the babies a quick lesson on colors and their names, then to reinforce the concept, you also run a tutorial on bucketing the screws. Slowly but surely, the infants begin sorting on their own. You’re hopeful that the babies will learn quickly despite their multitude of early mistakes. But alas, even after an hour of practice, the team is no where near ready.</p>



<p>Dismayed, you stop and step back. You conclude that whoever is helping you needs to understand colors. Although sorting the screws is a distinct problem from recognizing colors, the sorting step is trivial for someone who can identify colors. You send the babies back to their parents and recruit a couple of your slightly older cousins to help.</p>



<p>You effortlessly explain to the older children what you’re trying to accomplish, and the following tutorial proceeds smoothly this time. You’re astounded and you wonder why you didn’t follow this path from the beginning.</p>



<p>What you just did was the human equivalent of transfer learning. You took a trained brain—or stepping back from our analogy, a neural net—and you adapted it to a specialized problem. Transfer learning, or fine-tuning, is a process whereby you take a deep learning model that has been trained on lots of data (1M+ examples) and continue training it on a smaller dataset to “overfit” it to that particular class of problem. The model becomes inferior at its original task and better at the new specific task, but it also performs much better than a model that was only trained on the small problem-specific dataset.</p>



<p>Transfer learning is most commonly used in computer vision where most problems boil down to the analogous problem of detecting image features such as edges and shapes. A pretrained model—or one that has already been trained on a large dataset—has already learned all of the hard lessons and only needs to be adapted slightly to identify a new class of objects.</p>



<p>Under the hood, a neural network consists of a series of connected neurons and their weights. Neural net architectures, which define the ways in which the neural net’s neurons are connected, are always defined up-front, unlike the human brain which is elastic. But the neural net weights change through a process called backpropagation whereby the weights are updated based on mistakes that the network makes during training time. Returning to our color-screw analogy, this backpropagation process is analogous to you correcting the children when they make mistakes. </p>



<p>For computer vision problems, getting the neural net weights to be accurate enough for the model to detect anything takes millions of photos, but it’s easy to retrain the networks once they have learned the general concepts required to make useful inferences on photos.</p>



<p>In the industry, we often download models that have been pretrained on datasets like COCO and ImageNet and fine-tune them to our specific use case. At Passenger AI, for example, we use this process for object detection. </p>

