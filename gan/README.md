# Questions
1.  Why are we using Sequence-to-Sequence? This doesn't seem like the appropriate model
2.  Why is there an embedding layer? This seems unnecessary!


# PTB Word LM Questions
1.  Why do we need to pass the class is_training at initialization? This seems like a really odd abstraction, I'd rather define a model that can be passed to a training method.
2.  Why do people insist on putting embedding on the CPU?

# Python Questions
1.  What is @propetrty decorator and why are we using it?