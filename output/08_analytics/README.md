#Business Applications  
## Local Table of Contents  
[(Back to Master Table of Contents)](../)  
[Finance](#Finance)  
[A/B Testing](#A/BTesting)  
[Social Networks](#SocialNetworks)  
[Forecasting](#Forecasting)  
[Image Classification](#ImageClassification)  
[Document Classification](#DocumentClassification)  
[Spam Filter](#SpamFilter)  
[Sentiment Analysis](#SentimentAnalysis)  
## <a name="Finance"></a>Finance  

[An Introduction to Stock Market Data Analysis with Python ](https://ntguardian.wordpress.com/2016/09/19/introduction-stock-market-data-python-1/)  
by Curtis Miller  
"In these posts, I will discuss basics such as obtaining the data from Yahoo! Finance using pandas, visualizing stock data, moving averages, developing a moving-average crossover strategy, backtesting, and benchmarking. The final post will include practice problems. This first post discusses topics up to introducing moving averages." Part 2 is <a href='https://ntguardian.wordpress.com/2016/09/26/introduction-stock-market-data-python-2/'>here</a>.  
Other tags: [Tutorials in Python](../02_programming#TutorialsinPython)   
  
[Building a Financial Model with Pandas - Version 2](http://pbpython.com/amortization-model-revised.html)  
by Chris Mofitt  
Builds an amortization schedule in Pandas.  
Other tags: [Tutorials in Python](../02_programming#TutorialsinPython)   
  
## <a name="A/BTesting"></a>A/B Testing  

[A/B Testing with Hierarchical Models in Python](https://blog.dominodatalab.com/ab-testing-with-hierarchical-models-in-python/)  
by Manojit Nandi  
"In this post, I discuss a method for A/B testing using Beta-Binomial Hierarchical models to correct for a common pitfall when testing multiple hypotheses. I will compare it to the classical method of using Bernoulli models for p-value, and cover other advantages hierarchical models have over the classical model." Uses pymc in Python 2.  
Other tags: [Statistics Tutorials](../03_statistics#StatisticsTutorials), [Tutorials in Python](../02_programming#TutorialsinPython)   
  
[Analyze Your Experiment with a Multilevel Logistic Regression using PyMC3](https://dansaber.wordpress.com/2016/08/27/analyze-your-experiment-with-a-multilevel-logistic-regression-using-pymc3%E2%80%8B/)  
by Dan Saber  
Example of A/B testing with hierarchical models using pymc3.  
Other tags: [Statistics Tutorials](../03_statistics#StatisticsTutorials), [Tutorials in Python](../02_programming#TutorialsinPython)   
  
[A/B Testing Statistics](http://sl8r000.github.io/ab_testing_statistics/use_a_hierarchical_model/)  
by Slater Stich  
Demonstration of Hierarchical Models in A/B tests using pymc.  
Other tags: [Statistics Tutorials](../03_statistics#StatisticsTutorials), [Tutorials in Python](../02_programming#TutorialsinPython)   
  
## <a name="SocialNetworks"></a>Social Networks  

[Edge Prediction in a Social Graph: My Solution to Facebook's User Recommendation Contest on Kaggle](http://blog.echen.me/2012/07/31/edge-prediction-in-a-social-graph-my-solution-to-facebooks-user-recommendation-contest-on-kaggle/)  
by Edwin Chen  
"A couple weeks ago, Facebook launched a link prediction contest on Kaggle, with the goal of recommending missing edges in a social graph. I love investigating social networks, so I dug around a little, and since I did well enough to score one of the coveted prizes, I’ll share my approach here."  
Other tags: [Machine Learning Tutorials](../04_machine_learning#MachineLearningTutorials)   
  
## <a name="Forecasting"></a>Forecasting  

[Sorry ARIMA, but I’m Going Bayesian](http://multithreaded.stitchfix.com/blog/2016/04/21/forget-arima/)  
by Kim Larsen  
Tutorial on using Bayesian structural time series models.  
Other tags: [R Tutorials](../02_programming#RTutorials), [Statistics Tutorials](../03_statistics#StatisticsTutorials)   
  
## <a name="ImageClassification"></a>Image Classification  

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)  
by Stanford University  
"This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification." Course includes a <a href='http://cs231n.github.io/python-numpy-tutorial/'>Python tutorial.</a>  Uses Python 2.  
Other tags: [Deep Learning Lectures](../05_deep_learning#DeepLearningLectures), [Tutorials in Python](../02_programming#TutorialsinPython)   
  
[Not another MNIST tutorial with TensorFlow](https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow)  
by Justin Francis  
"Back when TensorFlow was released to the public in November 2015, I remember following TensorFlow’s beginner MNIST tutorial. I blindly copied and pasted all this code into my terminal and some numbers popped out as they should have. I thought, OK, I know there is something amazing happening here, why can I not see it? My goal was to make a MNIST tutorial that was both interactive and visual, and hopefully will teach you a thing or two that others just assume you know."  
Other tags: [Deep Learning Tutorials](../05_deep_learning#DeepLearningTutorials), [Tutorials in Python](../02_programming#TutorialsinPython)   
  
## <a name="DocumentClassification"></a>Document Classification  

[Data Science and (Unsupervised) Machine Learning with scikit-learn](http://opensource.datacratic.com/mtlpy50/)  
by Nicolas Kruchten  
"A different way to look at graph analysis and visualization, as an introduction to a few cool algorithms: Truncated SVD, K-Means and t-SNE with a practical walkthrough using scikit-learn and friends numpy and bokeh, and finishing off with some more general commentary on this approach to data analysis."  
Other tags: [Machine Learning Tutorials](../04_machine_learning#MachineLearningTutorials), [Python scikit-learn Tutorials](../02_programming#Pythonscikit-learnTutorials)   
  
## <a name="SpamFilter"></a>Spam Filter  

[Document Classification with scikit-learn](http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html)  
by Zac Stewart  
"To demonstrate text classification with scikit-learn, we're going to build a simple spam filter. While the filters in production for services like Gmail are vastly more sophisticated, the model we'll have by the end of this tutorial is effective, and surprisingly accurate."  
Other tags: [Python scikit-learn Tutorials](../02_programming#Pythonscikit-learnTutorials), [Machine Learning Tutorials](../04_machine_learning#MachineLearningTutorials)   
  
## <a name="SentimentAnalysis"></a>Sentiment Analysis  

[Out-of-core Learning and Model Persistence using scikit-learn](https://github.com/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/outofcore_modelpersistence.ipynb)  
by Sebastian Raschka  
"When we are applying machine learning algorithms to real-world applications, our computer hardware often still constitutes the major bottleneck of the learning process. Of course, we all have access to supercomputers, Amazon EC2, Apache Spark, etc. However, out-of-core learning via Stochastic Gradient Descent can still be attractive if we'd want to update our model on-the-fly ('online-learning'), and in this notebook, I want to provide some examples of how we can implement an 'out-of-core' approach using scikit-learn. I compiled the following code examples for personal reference, and I don't intend it to be a comprehensive reference for the underlying theory, but nonetheless, I decided to share it since it may be useful to one or the other!"  
Other tags: [Machine Learning Tutorials](../04_machine_learning#MachineLearningTutorials), [Python scikit-learn Tutorials](../02_programming#Pythonscikit-learnTutorials)   
  
