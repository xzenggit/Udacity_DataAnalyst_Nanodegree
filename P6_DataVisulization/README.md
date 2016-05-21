## Make effective data visulization

Create a data visualization from a data set that tells a story or highlights trends or patterns in the data. Use either dimple.js to create the visualization.

### Why this Project?

This project touches on the overarching attitudes and beliefs important to effective data visualization, such as:
* visualization is a dialog
* showcasing and sharing visualization with others
* visualization is a fluid process that typically requires multiple iterations of improvements

I experience the end-to-end process of creating effective data visualizations and highlighting important information from data that may otherwise be hidden or hard to uncover.

### What I learned

* Demonstrate the ability to choose optimal visual elements to encode data and critically assess the effectiveness of the visualization
* Communicate a story or finding to the appropriate audience using interactive visualizations
* Undergo the iterative process of creating a visualization, and build interactive visualizations with dimple.js or d3.js.

## Summary

This project explores factors that may affect whether a person can survive in the [Titanic event](https://www.kaggle.com/c/titanic). Different factors in the [Titanic dataset](https://www.kaggle.com/c/titanic/data) are analyzed, such as sex("Sex"), class level ("Pclass"), and embarked place("Embarked"). The analysis shows that generally people with the features of (female, class 1, and embarked from Cherburg) has a higher survival chance.

### Design

First, I download the [Titanic dataset](https://www.kaggle.com/c/titanic/data) and use the Ipyton Notebook do the exploratory analysis. Following the Kaggle instruction and tutorials, I start to explore the relation between "Sex" and "Survived". For visulization purpose, I add a "Count" column in the dataframe.

Afte that, I realized there are other factors also need to explore, such as age, class level, and embarked places, and number of relatives. Because there are many people with different ages, I add a new variable "Age_interval" to the data, and divide the origianl "Age" into 8 intervals from 0 to 80. 

Because "SibSp" and "Parch" are all faimily members, I add another new variable "Relatives_number" by adding "SibSp" and "Parch" together. I also changed the columna name "Survived" to "Status", and mapped the original 0/1 values to "Survived/Perished" for visulization purpose. With similar method, I looked into the differet values of "Pclass" and "Embarked" on the survival rate. Some variables have "NaN"s, which does give any information. So I exclude them during plotting.

As the reviewer suggested, I calcualted the survival rate for different categories. Because most variables are category data, I chose the bar plots for most figures. For different factors, I plotted the corresponding survival rates, which are easier to compare. 

## Feedback

* Feedback for index_v1.html: Visulization plot is good. But there are many other factors to be explored, such as age and class level. More explanations and data instruction should be added to make the reader understand better.

* Feedback for index_v2.html: The "Age" plot is very crowded, and doesn't provide much information. It would be great if it can be divided into different levels. It would be interesting to look into the interactions of these variables.

* Feedback for index_v2.html: The "False" and "True" categories for "Survived" are a little hard to understand. Suggest use "Survived" and "Perished" to for better understanding. Some categories has "NaN" also counted, which does give any useful information. 

* Feedback for index_v3.html: Some plots are too complex and do not show very obvious relations. Try to simplify the plots.

* Feedback for index_v4.html: Use percentage instead of point number. Set the y axis the same range. 

## Resources

* https://www.kaggle.com/c/titanic/data
* http://dimplejs.org
