# EECS-731-Project-7
EECS 731 Semester Project

## Abstract:
The articles were published by Mashable (www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. The original content is publicly accessed and retrieved using the provided URLs.
Acquisition date: January 8, 2015
The estimated relative performance values were estimated by the authors using a Random Forest classifier and rolling windows as assessment method.. See their article for more details on how the relative performance values were set.

## Attribute Information:
Number of Attributes: 61 (58 predictive attributes, 2 non-predictive, 1 goal field)

Attributes:

1. url: URL of the article (non-predictive)
2. timedelta: Days between the article publication and the dataset acquisition (non-predictive)
3. ntokenstitle: Number of words in the title
4. ntokenscontent: Number of words in the content
5. nuniquetokens: Rate of unique words in the content
6. nnonstop_words: Rate of non-stop words in the content
7. nnonstopuniquetokens: Rate of unique non-stop words in the content
8. num_hrefs: Number of links
9. numselfhrefs: Number of links to other articles published by Mashable
10. num_imgs: Number of images
11. num_videos: Number of videos
12. averagetokenlength: Average length of the words in the content
13. numkeywords: Number of keywords in the metadata 13. datachannelislifestyle: Is data channel 'Lifestyle'?
14. datachannelis_entertainment: Is data channel 'Entertainment'?
15. datachannelis_bus: Is data channel 'Business'?
16. datachannelis_socmed: Is data channel 'Social Media'?
17. datachannelis_tech: Is data channel 'Tech'?
18. datachannelis_world: Is data channel 'World'?
19. kwminmin: Worst keyword (min. shares)
20. kwmaxmin: Worst keyword (max. shares)
21. kwavgmin: Worst keyword (avg. shares)
22. kwminmax: Best keyword (min. shares)
23. kwmaxmax: Best keyword (max. shares)
24. kwavgmax: Best keyword (avg. shares)
25. kwminavg: Avg. keyword (min. shares)
26. kwmaxavg: Avg. keyword (max. shares)
27. kwavgavg: Avg. keyword (avg. shares)
28. selfreferencemin_shares: Min. shares of referenced articles in Mashable
29. selfreferencemax_shares: Max. shares of referenced articles in Mashable
30. selfreferenceavg_sharess: Avg. shares of referenced articles in Mashable
31. weekdayismonday: Was the article published on a Monday?
32. weekdayistuesday: Was the article published on a Tuesday?
33. weekdayiswednesday: Was the article published on a Wednesday?
34. weekdayisthursday: Was the article published on a Thursday?
35. weekdayisfriday: Was the article published on a Friday?
36. weekdayissaturday: Was the article published on a Saturday?
37. weekdayissunday: Was the article published on a Sunday?
38. is_weekend: Was the article published on the weekend?
39. LDA_00: Closeness to LDA topic 0
40. LDA_01: Closeness to LDA topic 1
41. LDA_02: Closeness to LDA topic 2
42. LDA_03: Closeness to LDA topic 3
43. LDA_04: Closeness to LDA topic 4
44. global_subjectivity: Text subjectivity
45. globalsentimentpolarity: Text sentiment polarity
46. globalratepositive_words: Rate of positive words in the content
47. globalratenegative_words: Rate of negative words in the content
48. ratepositivewords: Rate of positive words among non-neutral tokens
49. ratenegativewords: Rate of negative words among non-neutral tokens
50. avgpositivepolarity: Avg. polarity of positive words
51. minpositivepolarity: Min. polarity of positive words
52. maxpositivepolarity: Max. polarity of positive words
53. avgnegativepolarity: Avg. polarity of negative words
54. minnegativepolarity: Min. polarity of negative words
55. maxnegativepolarity: Max. polarity of negative words
56. title_subjectivity: Title subjectivity
57. titlesentimentpolarity: Title polarity
58. abstitlesubjectivity: Absolute subjectivity level
59. abstitlesentiment_polarity: Absolute polarity level
60. shares: Number of shares (target)

## Classification
1. Multinomial Naive Bayes classifier
2. Support Vector Machine Classifier
3. Random Forest Classifier

## Dimensionality reduction
1. Principal component analysis

## Clustering
1. Gaussian mixture Clustering 

## References:
1. https://www.kaggle.com/thehapyone/uci-online-news-popularity-data-set