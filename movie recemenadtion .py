#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("movie_metadata csv")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


# we have movies only upto 2016
import matplotlib.pyplot as plt
data.title_year.value_counts(dropna=False).sort_index().plot(kind='barh',figsize=(15,16))
plt.show()


# In[8]:


# recommendation will be based on these features only
data = data.loc[:,['director_name','actor_1_name','actor_2_name','actor_3_name','genres','movie_title']]


# In[9]:


data.head(10)


# In[10]:


data['actor_1_name'] = data['actor_1_name'].replace(np.nan, 'unknown')
data['actor_2_name'] = data['actor_2_name'].replace(np.nan, 'unknown')
data['actor_3_name'] = data['actor_3_name'].replace(np.nan, 'unknown')
data['director_name'] = data['director_name'].replace(np.nan, 'unknown')


# In[11]:


data


# In[12]:


data['genres'] = data['genres'].str.replace('|', ' ')


# In[13]:


data


# In[14]:


data['movie_title'] = data['movie_title'].str.lower()


# In[15]:


# null terminating char at the end
data['movie_title'][1]


# In[16]:


# removing the null terminating char at the end
data['movie_title'] = data['movie_title'].apply(lambda x : x[:-1])


# In[17]:


data['movie_title'][1]


# In[18]:


data.to_csv('data.csv',index=False)


# In[19]:


data.to_csv


# In[28]:


movie_title = 'Toy Story (1995)'
recommendations = (movie_title)
print(f'{movie_title}')


# In[41]:


trainset, testset = train_test_split(data, test_size=0.25)


# In[53]:


import numpy as np
import matplotlib.pyplot as plt

# Sample data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# Fit a linear regression model (best fit line)
coefficients = np.polyfit(x, y, 1)
best_fit_line = np.poly1d(coefficients)

# Generate y values for the best fit line
fit_line_y = best_fit_line(x)

# Plot the data points and the best fit line
plt.scatter(x, y, label='Data Points')
plt.plot(x, fit_line_y, color='red', label='Best Fit Line')

# Add labels and a legend
plt.xlabel('director')
plt.ylabel('rating')
plt.title('moive recemendation system')
plt.legend()

# Show the plot
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Sample data
categories = ['num_critic_for_reviews', 'duration']
values = [10, 20]

# Create a bar graph
plt.bar(categories, values, color='yellow')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('moivemetadata')

# Show the plot
plt.show()


# In[58]:


import matplotlib.pyplot as plt

# Sample data
labels = ['num_critic_for_reviews','duration']
sizes = [45, 55]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'blue', 'yellow'])

# Add a title
plt.title('Movie recemendation system')

# Show the plot
plt.show()


# In[ ]:




