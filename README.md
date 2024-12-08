# Cooking Up Calories: Predicting Recipe Nutrition with Data Science
## By Téa Hajratwala ([hajratwt@umich.edu](mailto:hajratwt@umich.edu)) and Devdeep Rajpal ([devdeep@umich.edu](mailto:devdeep@umich.edu))


### Introduction
We are using a portion of a dataset from food.com, containing data regarding each recipie posted to the site. Our truncated set starts from 2008 to current day. From the data, we focused on dealing with a few select portions of this dataset:
- Nutrition
  - Calories
  - Total Fat
  - Sugar
  - Sodium
  - Protein
  - Saturated Fat
  - Carbohydrates
- Minutes for each recipe
- Average Rating

We set out to try and predict the number of calories in a recipe based on the above factors, with the hope of using this predictor as part of analysis of recipes' health.

### Data Cleaning and Exploratory Analysis
#### Inital Data Cleaning
To clean this data for use, we took a handful of steps. We have a few of separate datasets, so we first had to merge a couple of them together for use. We then filled the ratings of zero with `np.nan` to allow our means to be accurate. As those means are calculated by `Pandas`, setting all 0 values to `np.nan` means they will be ignored.

We then took our third dataset, containing individual review data, and found the average rating for each recipe. With this, we then merged that data into our previous dataframe based on the recipe IDs. Now that all the data is in one place, we set the index of our data to those aformentioned IDs, and then made a new dataframe with all of thise data grouped by recipe ID.

Still, there was some work to do in individual columns. We took our nutrition data (mentioned in the first section), which was initally stored as a string of numbers, and turned it into a list, and then gave each value its own separate column. After adding this data to our main dataframe, we dropped the inital nutrition column and started looking at the data through some various styles of analyses.

<details>
  <summary style="color:blue;font-weight:bold;">Click To Open Table</summary>
<table>
<thead>
<tr>
<th style="text-align:right">recipe_id</th>
<th style="text-align:left">name</th>
<th style="text-align:right">id</th>
<th style="text-align:right">minutes</th>
<th style="text-align:right">contributor_id</th>
<th style="text-align:left">submitted</th>
<th style="text-align:left">tags</th>
<th style="text-align:right">n_steps</th>
<th style="text-align:left">steps</th>
<th style="text-align:left">description</th>
<th style="text-align:left">ingredients</th>
<th style="text-align:right">n_ingredients</th>
<th style="text-align:left">user_id</th>
<th style="text-align:left">date</th>
<th style="text-align:right">avg_rating</th>
<th style="text-align:left">review</th>
<th style="text-align:right">calories</th>
<th style="text-align:right">total_fat</th>
<th style="text-align:right">sugar</th>
<th style="text-align:right">sodium</th>
<th style="text-align:right">protein</th>
<th style="text-align:right">saturated_fat</th>
<th style="text-align:right">carbohydrates</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">275022</td>
<td style="text-align:left">impossible macaroni and cheese pie</td>
<td style="text-align:right">275022</td>
<td style="text-align:right">50</td>
<td style="text-align:right">531768</td>
<td style="text-align:left">2008-01-01</td>
<td style="text-align:left">[‘60-minutes-or-less’, ‘time-to-make’, ‘course’, ‘main-ingredient’, ‘preparation’, ‘main-dish’, ‘eggs-dairy’, ‘pasta’, ‘easy’, ‘cheese’, ‘dietary’, ‘high-calcium’, ‘high-in-something’, ‘pasta-rice-and-grains’, ‘elbow-macaroni’]</td>
<td style="text-align:right">11</td>
<td style="text-align:left">[‘heat oven to 400 degrees fahrenheit’, ‘grease a pie plate 10 x 1 1 / 2 inches’, ‘mix 2 cups cheese and the macaroni’, ‘sprinkle mixture in the plate’, ‘beat remaining ingredients , except the 1 / 4 cup cheese , until smooth , 15 seconds in a blender on high , or 1 minute with a hand beater’, ‘pour into plate’, ‘bake until knife inserted in center comes out clean - about 40 minutes’, ‘sprinkle with 1 / 4 cup cheese’, ‘bake until cheese is melted , 1 to 2 minutes’, ‘cool 10 minutes’, ‘serves 6 to 8 people’]</td>
<td style="text-align:left">one of my mom’s favorite bisquick recipes. this brings back memories!</td>
<td style="text-align:left">[‘cheddar cheese’, ‘macaroni’, ‘milk’, ‘eggs’, ‘bisquick’, ‘salt’, ‘red pepper sauce’]</td>
<td style="text-align:right">7</td>
<td style="text-align:left">[240552.0, 242794.0, 1221043.0]</td>
<td style="text-align:left">2008-04-07</td>
<td style="text-align:right">3</td>
<td style="text-align:left">[‘Easy comfort food! I definitely thought it was impossible, but it worked. 😄  I used 6 egg whites in place of the eggs and skim milk.  It came out super fluffy.  Thanks, Wendy!’, ‘I was looking for an easy Macaroni and Cheese for dinner.  My macaroni was hard and there were just too many eggs.  I think I might try again and precook the macaroni and only use 2 eggs.  It was just very disappointing.’, ‘Very easy recipe, nice presentation.  I used 2 cups  skim milk and 1/3 cup half and half.  Followed recipe exactly but left out hot sauce.  Macaroni was slightly al dente.  If you prefer well cooked macaroni, par boil it first for a few minutes.  Let baked pie cool for at least 15 minutes with foil covering the top.  This will allow the pie to &quot;settle&quot; so that it slices  into nicely formed wedges.  Definitely worth a try.  Am serving this with sauteed porto bello mushroms, sliced, on top of pie wedge, accompanied by a green salad.’]</td>
<td style="text-align:right">386.1</td>
<td style="text-align:right">34</td>
<td style="text-align:right">7</td>
<td style="text-align:right">24</td>
<td style="text-align:right">41</td>
<td style="text-align:right">62</td>
<td style="text-align:right">8</td>
</tr>
<tr>
<td style="text-align:right">275024</td>
<td style="text-align:left">impossible rhubarb pie</td>
<td style="text-align:right">275024</td>
<td style="text-align:right">55</td>
<td style="text-align:right">531768</td>
<td style="text-align:left">2008-01-01</td>
<td style="text-align:left">[‘60-minutes-or-less’, ‘time-to-make’, ‘course’, ‘preparation’, ‘healthy’, ‘pies-and-tarts’, ‘desserts’, ‘pies’, ‘dietary’]</td>
<td style="text-align:right">6</td>
<td style="text-align:left">[‘heat oven to 375 degrees’, ‘grease 10&quot; pan , put rhubarb in pan’, ‘blend all remaining ingredients for 3 minutes’, ‘pour over rhubarb’, ‘let set for a few minutes’, ‘bake 40 to 45 minutes’]</td>
<td style="text-align:left">a childhood favorite of mine. my mom loved it because it cut down on how much time to make it.</td>
<td style="text-align:left">[‘rhubarb’, ‘eggs’, ‘bisquick’, ‘butter’, ‘salt’, ‘sugar’, ‘vanilla’, ‘milk’]</td>
<td style="text-align:right">8</td>
<td style="text-align:left">[171303.0]</td>
<td style="text-align:left">2009-05-26</td>
<td style="text-align:right">3</td>
<td style="text-align:left">[‘When I found myself needing a dessert and having rhubarb on hand this recipe fit the bill.  I did however find it to be too overly sweet.  I would try it again, using less  sugar.  Thank you Starfire for sharing the recipe.’]</td>
<td style="text-align:right">377.1</td>
<td style="text-align:right">18</td>
<td style="text-align:right">208</td>
<td style="text-align:right">13</td>
<td style="text-align:right">13</td>
<td style="text-align:right">30</td>
<td style="text-align:right">20</td>
</tr>
<tr>
<td style="text-align:right">275026</td>
<td style="text-align:left">impossible seafood pie</td>
<td style="text-align:right">275026</td>
<td style="text-align:right">45</td>
<td style="text-align:right">531768</td>
<td style="text-align:left">2008-01-01</td>
<td style="text-align:left">[‘60-minutes-or-less’, ‘time-to-make’, ‘course’, ‘main-ingredient’, ‘preparation’, ‘very-low-carbs’, ‘main-dish’, ‘eggs-dairy’, ‘seafood’, ‘crab’, ‘cheese’, ‘dietary’, ‘low-sodium’, ‘low-calorie’, ‘low-carb’, ‘low-in-something’, ‘shellfish’]</td>
<td style="text-align:right">7</td>
<td style="text-align:left">[‘preheat oven to 400f’, ‘lightly grease large pie plate’, ‘mix crabmeat , cheeses and onion in pie plate’, ‘mix remaining ingredients in blender until smooth’, ‘slowly pour liquid mixture into pie plate’, ‘bake until golden brown for 35 to 40 minutes’, ‘let stand 5 minutes before cutting’]</td>
<td style="text-align:left">this is an oldie but a goodie. mom’s stand by for company. good enough for us on a special occasion or if company came over!</td>
<td style="text-align:left">[‘frozen crabmeat’, ‘sharp cheddar cheese’, ‘cream cheese’, ‘onion’, ‘milk’, ‘bisquick’, ‘eggs’, ‘salt’, ‘nutmeg’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[558429.0, 131804.0]</td>
<td style="text-align:left">2013-09-21</td>
<td style="text-align:right">3</td>
<td style="text-align:left">[‘Sorry, this one didn't work out so well. I did make some modifications that may have affected the taste of the recipe. I used imitation crabmeat, egg substitute, and fat free half-and-half. Unfortunately the end result was very bland–it definitely could have used some more spice. I really wanted to like this one but I won't be making this again. Thanks for posting anyway.’, ‘I have made this recipe for years, and we really love it, however, I do change it a bit. I use a whole 8 ounce package of cream cheese and just something like 3-4 oz of cheddar or American. We add some salt and let it cook only as long as it takes to puff up and then take it out to cool. Overcooked it becomes dry and less flavorful. We cook it in a Corning Ware &lt;br/&gt;9 inch quiche plate which holds all the ingredients just right. One of our favorite uses for imitation crab.’]</td>
<td style="text-align:right">326.6</td>
<td style="text-align:right">30</td>
<td style="text-align:right">12</td>
<td style="text-align:right">27</td>
<td style="text-align:right">37</td>
<td style="text-align:right">51</td>
<td style="text-align:right">5</td>
</tr>
<tr>
<td style="text-align:right">275030</td>
<td style="text-align:left">paula deen s caramel apple cheesecake</td>
<td style="text-align:right">275030</td>
<td style="text-align:right">45</td>
<td style="text-align:right">666723</td>
<td style="text-align:left">2008-01-01</td>
<td style="text-align:left">[‘60-minutes-or-less’, ‘time-to-make’, ‘course’, ‘preparation’, ‘occasion’, ‘desserts’, ‘cheesecake’, ‘gifts’, ‘taste-mood’, ‘sweet’]</td>
<td style="text-align:right">11</td>
<td style="text-align:left">[‘preheat oven to 350f’, ‘reserve 3 / 4 cup apple filling , and set aside’, ‘spoon remaining apple filling into the crust’, ‘beat together in large bowl , cream cheese , sugar , vanilla , eggs’, ‘pour over pie filling’, ‘bake for 35 minutes’, ‘cool’, ‘meanwhile , mix reserved pie filling and caramel topping and heat for 1 minute in a small saucepan’, ‘spread warm topping evenly over cooked , cooled cheesecake’, ‘decorate entire edge of cake with the 12 pecan halves , and sprinkle middle of cheesecake with chopped pecans’, ‘refrigerate , share , and enjoy !’]</td>
<td style="text-align:left">thank you paula deen!  hubby just happened to be watching with me one day when she made these and it will always be requested in our home!  it’s very easy to make and such a fun twist on a plain cheesecake.  it’s a must try!</td>
<td style="text-align:left">[‘apple pie filling’, ‘graham cracker crust’, ‘cream cheese’, ‘sugar’, ‘vanilla’, ‘eggs’, ‘caramel topping’, ‘pecan halves’, ‘pecans’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[156891.0, 276925.0, 723255.0, 55655.0, 437237.0, 951589.0, 739665.0, 115525.0, 231639.0, 2001170768.0]</td>
<td style="text-align:left">2008-01-18</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[“This was the first cheesecake I’d ever made.  It turned out great.  I substituted wheat free, gluten free spice cookies for the crust.  Thanks for this delicious recipe.”, “This has to be one of the best cheesecakes I’ve every had. It is so easy to make and my whole family loved it. Thanks Paula Deen!”, ‘All I can say about this is YUM!!  Has my two favorite desserts into one - apples and cheesecake.  It was also very easy to make which was a bonus.’, “Oh my!!! This is so good and very easy to make. I did everything according to the recipe, but used a dulce de leche instead of a caramel topping. It’s pretty much the same thing. It was heavenly. Thanks for posting.”, ‘I made this for our 16th wedding anniversary yesterday for dessert. YUMMY!! We all enjoyed this very much and I will most certainly make this over and over again. Thank you for a wonderful recipe.’, ‘AMAZING!!!  and soooo easy!!  I've never, ever finished a cheesecake this fast…I loved it! 😃  I only had a 9&quot; pie crust on hand, and some mini individual crusts- I was able to get make the 9&quot; and 3 minis off of the listed ingredients.  I loved not having to make my own crust in a springform pan like most cheesecakes call for.  Instead of putting pecan halves on top, I sprinkled chopped pecans around the rim of the cake…so pretty!  Thanks for a wonderful recipe, everyone LOVED it!’, ‘This is wonderful!!! So easy to make. My family loved it.’, “DH says this is the best thing he has ever eaten.  Enough said?  I love being how easy it is, and that it tastes so good.  I also love that you don’t need 2 pounds of cream cheese and a springform pan for it - you can have it anytime!  Next time I make it, I’ll chop up the apple pie filling.  I also added more pecans to the top.  Glad I picked up enough ingredients for 2 pies!”, ‘My daughter made this for us on Sun. IT was so good. WE loved it. She has made it before. It is always a hit.’, “This recipe is very easy and tasty. Not only have I done this exact recipe, I have exchanged it with my peach preserves that I spiced like I would a peach cobbler. There are so many variations of fillings you can do. I’ve experimented with many and my family and friends have loved them all!!”]</td>
<td style="text-align:right">577.7</td>
<td style="text-align:right">53</td>
<td style="text-align:right">149</td>
<td style="text-align:right">19</td>
<td style="text-align:right">14</td>
<td style="text-align:right">67</td>
<td style="text-align:right">21</td>
</tr>
<tr>
<td style="text-align:right">275032</td>
<td style="text-align:left">midori poached pears</td>
<td style="text-align:right">275032</td>
<td style="text-align:right">25</td>
<td style="text-align:right">307114</td>
<td style="text-align:left">2008-01-01</td>
<td style="text-align:left">[‘lactose’, ‘30-minutes-or-less’, ‘time-to-make’, ‘course’, ‘main-ingredient’, ‘cuisine’, ‘preparation’, ‘occasion’, ‘south-west-pacific’, ‘desserts’, ‘fruit’, ‘australian’, ‘easy’, ‘beginner-cook’, ‘dinner-party’, ‘summer’, ‘dietary’, ‘gluten-free’, ‘seasonal’, ‘egg-free’, ‘free-of-something’, ‘pears’, ‘taste-mood’, ‘sweet’]</td>
<td style="text-align:right">8</td>
<td style="text-align:left">[‘bring midori , sugar , spices , rinds and water to the boil’, ‘simmer for 5 minutes’, ‘peel the pears and remove the base end but leave the stem intact’, ‘place pears in hot liquid and simmer for approximately 15mins or until cooked’, ‘cooking time depends on how ripe the pears are’, ‘place each pear on a dessert plate’, ‘top each pear with 2 tablespoons reserved poaching liquid’, ‘garnish with orange rind curls and mint sprigs’]</td>
<td style="text-align:left">the green colour looks fabulous and the taste is heavenly. serve with a raspberry coulis. keep enough rind of the orange and lemon for garnish.</td>
<td style="text-align:left">[‘midori melon liqueur’, ‘water’, ‘caster sugar’, ‘cinnamon stick’, ‘vanilla pod’, ‘lemon rind’, ‘orange rind’, ‘pear’, ‘mint’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[306797.0]</td>
<td style="text-align:left">2008-03-21</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[‘This needs at least 10 stars.  The recipe was easy to make &amp; tasted magnificent.  I made a strawberry sauce that went wonderfully with it.  Thanks An_Net for sharing this keeper.’]</td>
<td style="text-align:right">386.9</td>
<td style="text-align:right">0</td>
<td style="text-align:right">347</td>
<td style="text-align:right">0</td>
<td style="text-align:right">1</td>
<td style="text-align:right">0</td>
<td style="text-align:right">33</td>
</tr>
</tbody>
</table>
</details>

#### Univariate Data Analysis
We took an inital look at preparation time.
<iframe src="univariate_analysis_prep_time.html" width="800" height="600" frameborder="0"></iframe>
You can see that the boxplot has some significant outliers-- There is one recipe that takes 1.05 _million_ minutes! We will need to remove these outliers to do most of the data analysis.

Let's take a look at the top outliers:
<details>
  <summary style="color:blue;font-weight:bold;">Click To Open Table</summary>
  <table>
<thead>
<tr>
<th style="text-align:right">recipe_id</th>
<th style="text-align:left">name</th>
<th style="text-align:right">id</th>
<th style="text-align:right">minutes</th>
<th style="text-align:right">contributor_id</th>
<th style="text-align:left">submitted</th>
<th style="text-align:left">tags</th>
<th style="text-align:right">n_steps</th>
<th style="text-align:left">steps</th>
<th style="text-align:left">description</th>
<th style="text-align:left">ingredients</th>
<th style="text-align:right">n_ingredients</th>
<th style="text-align:left">user_id</th>
<th style="text-align:left">date</th>
<th style="text-align:right">avg_rating</th>
<th style="text-align:left">review</th>
<th style="text-align:right">calories</th>
<th style="text-align:right">total_fat</th>
<th style="text-align:right">sugar</th>
<th style="text-align:right">sodium</th>
<th style="text-align:right">protein</th>
<th style="text-align:right">saturated_fat</th>
<th style="text-align:right">carbohydrates</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right">447963</td>
<td style="text-align:left">how to preserve a husband</td>
<td style="text-align:right">447963</td>
<td style="text-align:right">1051200</td>
<td style="text-align:right">576273</td>
<td style="text-align:left">2011-02-01</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘preparation’, ‘for-1-or-2’, ‘low-protein’, ‘5-ingredients-or-less’, ‘main-dish’, ‘1-day-or-more’, ‘easy’, ‘dietary’, ‘low-sodium’, ‘low-carb’, ‘low-in-something’, ‘number-of-servings’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[‘be careful in your selection’, “don’t choose too young”, ‘when selected , give your entire thoughts to preparation for domestic use’, ‘some wives insist upon keeping them in a pickle , others are constantly getting them into hot water’, ‘this may make them sour , hard and sometimes bitter’, ‘even poor varieties may be made sweet , tender and good by garnishing with patience , well sweetened with love and seasoned with kisses’, ‘wrap them in a mantle of charity’, ‘keep warm with a steady fire of domestic devotion and serve with peaches and cream’, ‘thus prepared , they will keep for years’]</td>
<td style="text-align:left">found this in a local wyoming cookbook “a collection of recipes using wine, cordials, and beer” from the broadway liquor store by sara gradin (by the way this particular cookbook was typed up and price on cover was .50 cents!!)</td>
<td style="text-align:left">[‘cream’, ‘peach’]</td>
<td style="text-align:right">2</td>
<td style="text-align:left">[526666.0, 1072593.0]</td>
<td style="text-align:left">2011-03-10</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[“I’d thought that I would like to keep mine in vinegar, as that is a good preservative, but that makes him too sour. Then, I considered keeping him filled with alcohol, as I know that is another type of preservative, but, who wants a sloshed hubby all the time? Not me! And he’s way too big for the jars that I have around the house. I believe that smothering him with love and kindness will get me the furthest in the ‘keeping him’ stage. And none of that ‘putting him on a shelf’ stuff for me; he deserves to be seen and admired! Thanks so much, Chef GreanEyes, for sharing such a thought-provoking recipe! 😉”, “No matter if you’ve got the basic, no-frills model or one like Donald Trump and his billions, I know this technique to be true.  I’ll spread the word.”]</td>
<td style="text-align:right">407.4</td>
<td style="text-align:right">57</td>
<td style="text-align:right">50</td>
<td style="text-align:right">1</td>
<td style="text-align:right">7</td>
<td style="text-align:right">115</td>
<td style="text-align:right">5</td>
</tr>
<tr>
<td style="text-align:right">291571</td>
<td style="text-align:left">homemade fruit liquers</td>
<td style="text-align:right">291571</td>
<td style="text-align:right">288000</td>
<td style="text-align:right">553251</td>
<td style="text-align:left">2008-03-12</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘main-ingredient’, ‘preparation’, ‘occasion’, ‘5-ingredients-or-less’, ‘beverages’, ‘desserts’, ‘fruit’, ‘1-day-or-more’, ‘easy’, ‘dinner-party’, ‘cocktails’, ‘berries’]</td>
<td style="text-align:right">12</td>
<td style="text-align:left">[‘rinse the fruit or berries , fruit must be cut into small pieces’, ‘place berries or fruit in a container , add vodka’, ‘cap and store in a cool , dark place , stir once a week for 2 - 4 weeks’, ‘strain through metal colander’, ‘transfer the unsweetened liqueur to an aging container’, ‘to 3 cups ml unsweetened liqueur add 1 1 / 4 cup granulated sugar’, ‘let age for at least three months’, ‘pour carefully the clear liqueur to a new bottle’, ‘add more sugar if necessary’, ‘the flavor of almost all liqueurs improves during storage’, ‘fruit and berry liqueurs should be stored for at least 6 months for maximum taste’, ‘some lemon liqueurs should not be stored for a long time’]</td>
<td style="text-align:left">this should be a nice easy project for those willing to wait and enjoy things made by hand.  fruit liqueurs can be used anywhere from beverages to dessert and hopefully it won’t become addictive from shear flavor alone. apricots, blackberries, black currants, blueberries, cherries, cranberries, nectarines, peaches, plums and/or raspberries should be used to give the best liqueur.</td>
<td style="text-align:left">[‘berries’, ‘vodka’, ‘granulated sugar’]</td>
<td style="text-align:right">3</td>
<td style="text-align:left">[511661.0]</td>
<td style="text-align:left">2008-03-13</td>
<td style="text-align:right">4</td>
<td style="text-align:left">[‘Thanks for the extra tip about citrus liquers not lasting well- in the past, I wondered why mine turned bitter!  A tip, you can add 1/4 to 1/2 tsp food-grade glycerin for smoothness once the liquer has aged.’]</td>
<td style="text-align:right">836.2</td>
<td style="text-align:right">0</td>
<td style="text-align:right">333</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">27</td>
</tr>
<tr>
<td style="text-align:right">425681</td>
<td style="text-align:left">homemade vanilla</td>
<td style="text-align:right">425681</td>
<td style="text-align:right">259205</td>
<td style="text-align:right">28177</td>
<td style="text-align:left">2010-05-16</td>
<td style="text-align:left">[‘time-to-make’, ‘preparation’, ‘5-ingredients-or-less’, ‘1-day-or-more’, ‘easy’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[‘slice the vanilla beans length-wise and scrape the seeds out’, ‘cut the empty pods into 1-2 inch pieces and add both the seeds and pods to a one pint glass jar’, ‘add vodka’, ‘seal jar with canning lid and ring or cap and shake vigorously’, ‘label and date the jar with the start date and the end date , six months from today’, ‘place jar in a cupboard , away from sunlight’, ‘shake jar once a day for a week , then once a week for a couple months , then once a month until mature’, ‘after 6 months you can strain out the extract , leaving any seeds &amp; pods in the jar and start a second batch’, ‘the second batch may take longer to mature’]</td>
<td style="text-align:left">found this recipe on tammy’s blog (<a href="http://www.tammysrecipes.com/homemade_vanilla_extract">http://www.tammysrecipes.com/homemade_vanilla_extract</a>) and couldn’t resist trying it.  i made 3 batches; one with vodka, one with gluten-free vodka and one with bourbon.  it’ll be 6 months before they’re totally mature, but i might have to test one out a little sooner.  😃  i purchased 1/2 lb of very fresh madagascar bourbon vanilla beans online for this recipe.  the recipe is for 12 ozs of vanilla essence but it can be made in any amount needed.  just figure out how many ounces you want to make and add more vanilla beans in the same ratio.</td>
<td style="text-align:left">[‘vanilla beans’, ‘vodka’]</td>
<td style="text-align:right">2</td>
<td style="text-align:left">[471004.0]</td>
<td style="text-align:left">2011-01-14</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[“This is how I have been making vanilla for several years.  That is when I don’t cheat &amp; buy it from Costco.  I usually only do a 6 oz batch with 1 vanilla bean.  If you do a lot of baking you can start 3 separate batches a couple of months apart to have a steady supply of vanilla.  Don’t forget you can reuse the vanilla beans for a 2nd or even 3rd batch of vanilla, just cut back on the amount of vodka you add.  The hardest part is waiting for the vanilla to mature.  Thanks for posting this Tink.”]</td>
<td style="text-align:right">69.4</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
</tr>
<tr>
<td style="text-align:right">463624</td>
<td style="text-align:left">homemade vanilla extract</td>
<td style="text-align:right">463624</td>
<td style="text-align:right">129600</td>
<td style="text-align:right">1722785</td>
<td style="text-align:left">2011-09-05</td>
<td style="text-align:left">[‘time-to-make’, ‘preparation’, ‘occasion’, ‘for-large-groups’, ‘5-ingredients-or-less’, ‘1-day-or-more’, ‘easy’, ‘gifts’, ‘oamc-freezer-make-ahead’, ‘inexpensive’, ‘number-of-servings’, ‘from-scratch’]</td>
<td style="text-align:right">12</td>
<td style="text-align:left">[‘carefully open the bottle of brandy’, ‘pour off approximately one shot of liquid’, ‘on a small cutting board slice vanilla beans in half , then again length wise’, ‘this will produce 4 strips of vanilla bean husk per each original bean’, ‘with a sharp fine’, ‘put all gathered vanilla pulp straight into bottle of brandy , and then place the husk inside the bottle as well’, ‘repeat until all slices of bean husk have been stripped and dropped into the bottle’, ‘re-cork / close bottle and gently shake to disburse the vanilla husks and pulp in the liquid’, ‘this is a very slow steeping process so do not rush the immersion’, ‘store bottle on in a space where you will remember to stir / swirl the bottle aprox once a week for at least 2 to 3 months’, ‘when you pour off some of your homemade vanilla into a smaller bottle for easier usage , remember to refill the original bottle back up with fresh alcohol , and begin the immersion process again’, ‘the original beans will last at several steeps , in fact my original vanilla has been used for 4 batches now’]</td>
<td style="text-align:left">after getting a very poor bottle of vanilla extract from the “fine goods” grocery store a couple years ago i researched how to make my own vanilla from scratch. you may use several different kinds of alochol but i much prefer the heavier sweeter brandy to the more commonly used rum or vodka. i specifically use the korbel xs because in the distilling process they use extra vanilla and spices in their recipe.</td>
<td style="text-align:left">[‘brandy’, ‘vanilla beans’]</td>
<td style="text-align:right">2</td>
<td style="text-align:left">[186851.0]</td>
<td style="text-align:left">2011-09-06</td>
<td style="text-align:right">nan</td>
<td style="text-align:left">[“I love this vanilla! I have been making vanilla this way for about 3 years now. The only time I don’t use this vanilla, is when I am making something for children.”]</td>
<td style="text-align:right">75.2</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
</tr>
<tr>
<td style="text-align:right">435928</td>
<td style="text-align:left">peach cordial</td>
<td style="text-align:right">435928</td>
<td style="text-align:right">86415</td>
<td style="text-align:right">597669</td>
<td style="text-align:left">2010-08-24</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘main-ingredient’, ‘preparation’, ‘for-large-groups’, ‘beverages’, ‘fruit’, ‘1-day-or-more’, ‘easy’, ‘number-of-servings’, ‘3-steps-or-less’]</td>
<td style="text-align:right">7</td>
<td style="text-align:left">[‘in a gallon glass mayonnaise jar , or other type of crock that can be sealed completely , mix all the ingredients and seal with electrical or duct tape so no aid can get in or liquid can evaporate’, ‘shake gently twice a day until the sugar is dissolved then about every other day so that the mixture is evenly flavored’, ‘now the hard part – let sit , for 2 months’, ‘then strain through 4 layers of cheesecloth in a colander’, ‘after sediment has settled , siphon off the clear cordial mixture and decant’, ‘this is really smooth and friends are still asking if i have the recipe’, ‘i hope that you enjoy it’]</td>
<td style="text-align:left">now that peach season is here, make some of this delicious cordial to save for when the cooler temps come along.  this will also make great gifts.  after straining off the cordial itself, do not dump the peaches.  pick out the flavoring spices and run through a food mill or blender and serve over ice cream.  please note that the cooking time is the time it takes for the peaches to flavor the vodka.</td>
<td style="text-align:left">[‘peaches’, ‘granulated sugar’, ‘cinnamon sticks’, ‘lemon peel’, ‘whole cloves’, ‘vodka’]</td>
<td style="text-align:right">6</td>
<td style="text-align:left">[1803366096.0, 2000422937.0, 2001654933.0]</td>
<td style="text-align:left">2014-11-18</td>
<td style="text-align:right">4.5</td>
<td style="text-align:left">[‘This turned out well. We'll try it again but will halve the sugar as it's generally too sweet for us. I think most everyone will enjoy it with the original amount of sugar called for. We added grated fresh ginger to the spice compliment, 6 tablespoons. Yum!’, ‘How should I prepare the peaches - should they be cut into pieces? Should I remove the pits? Also, what about the peel?’, ‘I am surprised that your mayonnaise jars did not explode given that there are gases that build up and need to be vented. I used a brewing carboy with an airlock. I added the spices as noted. Waiting to see how it tastes in another month.’]</td>
<td style="text-align:right">111.2</td>
<td style="text-align:right">0</td>
<td style="text-align:right">46</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">3</td>
</tr>
<tr>
<td style="text-align:right">372282</td>
<td style="text-align:left">chocolate chunk vanilla cake</td>
<td style="text-align:right">372282</td>
<td style="text-align:right">72000</td>
<td style="text-align:right">883095</td>
<td style="text-align:left">2009-05-16</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘preparation’, ‘for-large-groups’, ‘desserts’, ‘1-day-or-more’, ‘cakes’, ‘number-of-servings’]</td>
<td style="text-align:right">10</td>
<td style="text-align:left">[‘preheat oven to 350f grease an 8 or 9-inch square baking dish’, ‘cream butter and sugar until light and fluffy’, ‘add eggs one at a time , beating well after each addition’, ‘beat in vanilla’, ‘mix dry ingredients together’, ‘add half of dry mixture to wet ingredients’, ‘add carnation milk and then remaining dry mixture’, ‘add chopped chocolate’, ‘spoon batter into prepared pan and spread evenly’, ‘bake 45-50 minutes until a toothpick inserted in center comes out clean’]</td>
<td style="text-align:left">this quick no fuss cake travel very well, so it’s a great one to make for those pot-luck family affairs.</td>
<td style="text-align:left">[‘butter’, ‘granulated sugar’, ‘eggs’, ‘vanilla’, ‘cake-and-pastry flour’, ‘baking powder’, ‘salt’, ‘carnation evaporated milk’, ‘semisweet chocolate’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[881977.0]</td>
<td style="text-align:left">2009-07-01</td>
<td style="text-align:right">4</td>
<td style="text-align:left">[“This was quite good!  My son decided he wanted something different for his birthday, so we made this cake.  He loved it and has already asked when we can make it again.  It’s an extremely rich cake, so small pieces are in order, with ice cream or milk to balance it out.”]</td>
<td style="text-align:right">233.8</td>
<td style="text-align:right">18</td>
<td style="text-align:right">50</td>
<td style="text-align:right">5</td>
<td style="text-align:right">8</td>
<td style="text-align:right">36</td>
<td style="text-align:right">10</td>
</tr>
<tr>
<td style="text-align:right">479702</td>
<td style="text-align:left">flavored vinegar</td>
<td style="text-align:right">479702</td>
<td style="text-align:right">64815</td>
<td style="text-align:right">1195537</td>
<td style="text-align:left">2012-05-21</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘cuisine’, ‘preparation’, ‘occasion’, ‘for-large-groups’, ‘condiments-etc’, ‘french’, ‘1-day-or-more’, ‘easy’, ‘european’, ‘dinner-party’, ‘heirloom-historical’, ‘vegetarian’, ‘marinades-and-rubs’, ‘dietary’, ‘gifts’, ‘oamc-freezer-make-ahead’, ‘inexpensive’, ‘number-of-servings’]</td>
<td style="text-align:right">7</td>
<td style="text-align:left">[‘collect the number of bottles necessary , with sound corks to fit’, ‘wash the bottles in soapy water , rinse first in very hot then in cold water , drain , dry and heat in a slow oven’, ‘scald the corks in boiling water’, ‘pour vinegar into an enamel lined or stainless steel pan and over low temperature slowly heat , do not let boil’, ‘add shallots , garlic , seeds and / or sprigs of herbs to the warm bottle’, ‘if using tarragon , use a long sprig , twice the hight of the bottle , bend it double and push it down the neck of the bottle’, ‘fill up with warm vinegar , cork down tightly , place on a sunny window sill to mature for 4 - 6 weeks before using’]</td>
<td style="text-align:left">adapted from the book “the french farmhouse kitchen” by eileen reece.</td>
<td style="text-align:left">[‘white vinegar’, ‘shallots’, ‘garlic cloves’, ‘raspberries’, ‘mustard seeds’, ‘dill seeds’, ‘juniper berries’, ‘rosemary’, ‘tarragon’]</td>
<td style="text-align:right">9</td>
<td style="text-align:left">[471004.0]</td>
<td style="text-align:left">2016-05-01</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[‘Sounds like some great flavour combinations. A couple of tips, the best info I found on making fruit infused vinegar said to use equal weights of vinegar &amp; fruit, let it infuse in a cool dark place for a minimum of 6 weeks. I would think this would be ideal for herb vinegars as well to give you a more intense flavour. We actually preferred using apple cider vinegar rather than wine vinegar for most of the infused vinegars I made. It seems to be a mellower finish. Now from experience, make your vinegars in a mason jar rather than the bottle you will gift them in. You need to strain your vinegar after infusing. Once it is strained you can decant it into the bottles you are using for gifting.’]</td>
<td style="text-align:right">15.4</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
</tr>
<tr>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left">flavored wine vinegar has been an important ingredient in french cooking since medieval times when vinegar was essential in order to keep meat edible in warm weather. use whatever herbs and seeds you like or have in the garden. thread perl onions or garlic with a darning needle on a fine string, tie around the cork and suspend into the vinegar.</td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
</tr>
<tr>
<td style="text-align:right">437385</td>
<td style="text-align:left">sauerkraut in a bottle</td>
<td style="text-align:right">437385</td>
<td style="text-align:right">60510</td>
<td style="text-align:right">1129191</td>
<td style="text-align:left">2010-09-14</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘main-ingredient’, ‘cuisine’, ‘preparation’, ‘low-protein’, ‘healthy’, ‘5-ingredients-or-less’, ‘condiments-etc’, ‘vegetables’, ‘german’, ‘1-day-or-more’, ‘easy’, ‘european’, ‘low-fat’, ‘vegetarian’, ‘dietary’, ‘low-cholesterol’, ‘low-saturated-fat’, ‘low-calorie’, ‘low-carb’, ‘low-in-something’, ‘from-scratch’]</td>
<td style="text-align:right">18</td>
<td style="text-align:left">[‘1’, ‘quarter , core , and shred cabbage’, ‘2’, ‘pack into sterilized quart jars by tamping down with a fork and leave 1 inch headspace’, ‘3’, ‘add 2 tsp salt and 3 tsp cider vinegar to each jar’, ‘4’, ‘cover with boiling water to within 1 / 2 an inch of the rim , pouring slowly and working air bubbles out with a fork’, ‘5’, ‘cover with standard self-sealing lids’, ‘6’, ‘apply bands firmly’, ‘7’, ‘turn upside down on a tea towel for a day’, ‘check seals after 24 hours’, ‘8’, ‘store in a cool , dark place and let cure for 6 weeks’, ‘9’]</td>
<td style="text-align:left">if you love homemade sauerkraut, but don’t have your own crocks, this is a great recipe. it’s fast and easy to put together and you can make any quantity you want. my father made sauerkraut every year with friends, since he doesn’t have crocks, and another friend gave him this recipe to make life easier.</td>
<td style="text-align:left">[‘cabbage’, ‘salt’, ‘cider vinegar’, ‘boiling water’]</td>
<td style="text-align:right">4</td>
<td style="text-align:left">[1072593.0, 1803028572.0]</td>
<td style="text-align:left">2011-10-05</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[“Ach du lieber!  Does sauerkraut not come in a clear glass jar off the grocer’s shelf any longer?  A dying art.  Made for PAC Fall 2011.”, ‘No need to add any vinegar!  Salt alone preserves sour cabbage very well.  I add 1/4 cup of shredded carrot for colour.  Sometimes, I add a laurel leaf and 1/2 teaspoon of caraway seeds.’]</td>
<td style="text-align:right">18.3</td>
<td style="text-align:right">0</td>
<td style="text-align:right">9</td>
<td style="text-align:right">48</td>
<td style="text-align:right">1</td>
<td style="text-align:right">0</td>
<td style="text-align:right">1</td>
</tr>
<tr>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left">note: cooking time is time to cure.</td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
</tr>
<tr>
<td style="text-align:right">437520</td>
<td style="text-align:left">simple hard apple cider</td>
<td style="text-align:right">437520</td>
<td style="text-align:right">53290</td>
<td style="text-align:right">833434</td>
<td style="text-align:left">2010-09-16</td>
<td style="text-align:left">[‘time-to-make’, ‘course’, ‘main-ingredient’, ‘cuisine’, ‘preparation’, ‘occasion’, ‘north-american’, ‘5-ingredients-or-less’, ‘beverages’, ‘fruit’, ‘american’, ‘canadian’, ‘1-day-or-more’, ‘easy’, ‘beginner-cook’, ‘holiday-event’, ‘kosher’, ‘vegan’, ‘vegetarian’, ‘cocktails’, ‘punch’, ‘dietary’, ‘oamc-freezer-make-ahead’, ‘apples’, ‘heirloom-historical-recipes’, ‘from-scratch’]</td>
<td style="text-align:right">14</td>
<td style="text-align:left">[“pour out a little of the juice so there’s room , pour the yeast and 1 1 / 2 cups sugar in , top back up with apple juice but leave at least an inch of room at the top”, ‘replace the lid and shake well to dissolve yeast and sugar as much as possible’, ‘stretch a balloon over the top of the jug instead of the lid , poke a few holes in the top of the balloon with a pin’, ‘this is your air lock’, ‘it will allow co2 to escape without letting bacteria inches i like to wrap string or a rubber band around the balloon to keep it firmly in place and sealed’, ‘set in a closet or basement and ignore for a month’, ‘when you come back your balloon will be somewhat limp and your brew should be back to as clear as it was before you added things’, ‘if you used store brand apple juice it should be clear , if organic it will still be somewhat cloudy , either way there should be a clear separation from the sediment at the bottom’, ‘either carefully pour out your cider , leaving the sediment at the bottom , or siphon out’, “add as much of the remaining sugar as pleases your taste buds , at least 1 / 4 cup though or it won’t carbonate”, ‘wash out some pop bottles really well , this looks ghetto but they were designed to handle the pressure of carbonation and wine bottles could explode’, “if you want to be really pro you can get proper beer bottles and stuff , i just don’t serve in these”, “pour your brew into the bottles , try not to leave them half full , you’ll need one two liter and one 500ml bottle , unless you spill a bunch while siphoning , like i did”, “seal the lids on well and let sit for another week to mellow and carbonate , chill and drink ! it will mellow more as it ages but i wouldn’t let it go more then a year”]</td>
<td style="text-align:left">this turned out dangerously tasty! i think i’m going to make a 5 gallon batch. cooking time is ignore-it-in-the-closet time. making very tasty booze was never this easy!</td>
<td style="text-align:left">[‘apple juice’, ‘champagne yeast’, ‘sugar’]</td>
<td style="text-align:right">3</td>
<td style="text-align:left">[2000842620.0]</td>
<td style="text-align:left">2016-03-02</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[‘This was so easy, and it turned out great!  It's even great without carbonation (I got impatient before it had the chance to carbonate).’]</td>
<td style="text-align:right">375.4</td>
<td style="text-align:right">0</td>
<td style="text-align:right">368</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">0</td>
<td style="text-align:right">31</td>
</tr>
<tr>
<td style="text-align:right">309383</td>
<td style="text-align:left">pickled olives</td>
<td style="text-align:right">309383</td>
<td style="text-align:right">50410</td>
<td style="text-align:right">60124</td>
<td style="text-align:left">2008-06-15</td>
<td style="text-align:left">[‘time-to-make’, ‘preparation’, ‘low-protein’, ‘5-ingredients-or-less’, ‘very-low-carbs’, ‘1-day-or-more’, ‘easy’, ‘dietary’, ‘low-cholesterol’, ‘low-calorie’, ‘low-carb’, ‘low-in-something’]</td>
<td style="text-align:right">12</td>
<td style="text-align:left">[‘pick over the olives , discard any with big blemishes’, ‘with a parring knife , cut down the side of the olive , thru to the stone , then turn over and repeat on the other side’, ‘place the olives in a 2 litre sterilized jars , untill the jars are 2 / 3 full , then cover olives with cold water’, ‘to keep the olives submerged , fill a small plastic bag with water , and sit it on top of the olives’, “change the water daily , scum may appear on the surface , but that’s fine”, ‘change the water for 4 days with black olives , and for 6 days with green olives’, ‘combine the salt and water , stir over heat until the salt has dissolved , cool’, ‘drain and discard the water in the jars , fill with enough salted water to cover the olives’, ‘pour enough oil into the jars to cover the olives completely , and then seal the jars’, ‘mark the date on the jars and store in a cool dark place for 5 weeks’, ‘after 5 weeks they are ready to eat , but you can then marinate them for another 2 weeks using lemon wedges and garlic , or whatever you like’, ‘cover with oil’]</td>
<td style="text-align:left">i searched everywhere for this recipe!! a friend gave me a bag of green olives, because i had said i would like to try pickling my own.</td>
<td style="text-align:left">[‘green olives’, ‘fine sea salt’, ‘water’, ‘olive oil’]</td>
<td style="text-align:right">4</td>
<td style="text-align:left">[29196.0]</td>
<td style="text-align:left">2012-07-21</td>
<td style="text-align:right">5</td>
<td style="text-align:left">[“OMG, OMG. OMG we’ve been eating these for days at mummamills home. We’ve been stranded here while our car needs to be fixed, so we keep eating!!!”]</td>
<td style="text-align:right">313</td>
<td style="text-align:right">51</td>
<td style="text-align:right">3</td>
<td style="text-align:right">254</td>
<td style="text-align:right">3</td>
<td style="text-align:right">22</td>
<td style="text-align:right">1</td>
</tr>
<tr>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left">i love the whole greek thing, making your own cheese and all that.</td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
</tr>
<tr>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left">anyway, i finally found this recipe in a womens weekly book. putting it here so i don’t have to search!</td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
</tr>
<tr>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left">it is excellent, better then the bought one 😃</td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:left"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
<td style="text-align:right"></td>
  </tr>
  </tbody>
  </table>
</details>

It appears that these are mostly pickling/marinating/fermenting recipes, which take a long time. Thankfully, the `tags` column contains a tag that labels recipes that take `1-day-or-more`-- We can filter those out.
<iframe
  src="univariate_analysis_prep_time_cleaned.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
We also took a look at the distributions of average ratings, and it appears that the vast majority of average ratings are positive. Even when increasing the histogram bin size (and increasing granularity), the rightmost bin consistently has the highest number of recipes.
<iframe
  src="univariate_analysis_rating_dist.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
#### Bivariate Data Analysis
We took a look at if there's any relation between the average user rating and prep time in a recipe, and found no great correlation. On the other hand, taking a look at the average protein to carb ration for many of the common tags did show some interesting results.
<iframe
  src="bivariate_analysis_protein_carb_to_tags.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
You can see (not surprisingly) that the recipes with the highest average protein to carb ratio are `low-carb` and `meat` recipes. Surprisingly, recipes labelled as `healthy` have some of the lowest protein to carb ratios in the top 25 most common tags.

#### Interesting Aggregates

Another way of visuallizing the data above in a less condensed way is with a grouped table:

|   calories |   total_fat |    sugar |   sodium |   protein |   saturated_fat |   carbohydrates |
|-----------:|------------:|---------:|---------:|----------:|----------------:|----------------:|
|    313.673 |     24.2679 |  67.9461 | 26.5028  |   17.3269 |         28.1758 |        10.3824  |
|    358.428 |     26.5037 |  63.8623 | 28.428   |   25.8313 |         31.0342 |        11.7752  |
|    375.627 |     28.58   |  48.9627 | 23.481   |   31.692  |         34.2357 |        11.6247  |
|    565.994 |     42.9425 |  93.5872 | 36.4806  |   42.7273 |         54.3618 |        18.6147  |
|    445.946 |     33.7911 |  66.5634 | 26.9887  |   34.7885 |         42.8807 |        14.4488  |
|    428.778 |     32.4828 |  69.662  | 28.4942  |   32.3909 |         40.2891 |        13.8916  |
|    440.97  |     34.444  |  58.331  | 32.0511  |   36.1374 |         40.9731 |        13.2512  |
|    512.356 |     38.5901 | 175.292  | 12.9782  |   13.7135 |         59.1052 |        21.8914  |
|    408.545 |     29.9295 |  68.2619 | 27.4942  |   31.3608 |         36.4003 |        13.7477  |
|    384.713 |     28.7112 |  65.5739 | 28.3566  |   29.0385 |         34.9496 |        12.5277  |
|    355.049 |     11.392  |  90.698  | 30.3171  |   26.5765 |         11.9766 |        18.8234  |
|    380.615 |     36.2018 |  24.1802 | 31.8497  |   45.4427 |         42.2459 |         5.39205 |
|    316.321 |     12.8514 |  77.5287 | 31.9126  |   20.4518 |         10.3713 |        16.1364  |
|    411.148 |     30.1459 |  71.6231 | 28.6162  |   33.4537 |         36.1988 |        13.3935  |
|    401.297 |     27.2916 |  94.8222 |  7.27786 |   27.0554 |         34.1021 |        15.032   |
|    504.188 |     39.7138 |  30.9946 | 37.4495  |   59.9219 |         46.7332 |        11.6843  |
|    426.742 |     33.0329 |  55.3966 | 28.242   |   38.0331 |         40.2653 |        12.5426  |
|    522.895 |     43.9799 |  33.01   | 40.5964  |   65.34   |         50.5548 |        10.0593  |
|    448.028 |     35.125  |  68.4093 | 32.6948  |   35.681  |         43.8655 |        13.6219  |
|    361.98  |     26.777  |  65.2238 | 24.0574  |   25.2343 |         32.7225 |        11.8512  |
|    436.785 |     33.6179 |  72.2855 | 27.8589  |   32.568  |         42.0059 |        13.9585  |
|    428.758 |     32.5336 |  68.5713 | 28.7823  |   32.9424 |         40.1143 |        13.77    |
|    453.661 |     35.1819 |  67.2312 | 28.5374  |   35.8446 |         44.8645 |        14.103   |
|    427.361 |     32.4837 |  69.0974 | 28.3956  |   32.0823 |         40.0908 |        13.8383  |
|    336.174 |     25.9753 |  34.8879 | 25.699   |   26.1295 |         29.4537 |        10.7623  |

Here, we have left the data in its raw form (ie. not in ratios).

#### Imputation

We have imputed the data to *remove* missing data. We did this because ultimately the imputed dataset is still large enough to be representative of the raw dataset, and ultimatlely it doesn't affect the appearance or shape of the data. See below:

`recipes_non_imputed.shape`

(1364760, 8)

`recipes_imputed = recipes_imputed.dropna()`

`recipes_imputed.shape`

(1364760, 8)

When examining only nutritional values, the imputed dataset has the same shape as the non-imputed dataset. So in this case, dropping NA values is sufficient (or not doing imputation at all).

### Framing a Prediction Problem

Our question is as such: when given a vector of nutritional values, the number of minutes needed to finish the recipe, and the average rating, what would be the predicted number of calories?

This question is a regression problem, since the `calories` column is numerical. We decided to classify this variable because it is easliy interpretable for someone looking to make something with certain caloric needs. Our metric for success will be mean squared error, as it is the easiest to calculate (although in ridge regression this is not what is minimized). However, this approximation should be enough to estimate the success of the model.

### Baseline Model
The model used is a ridge regression, with a pipline that includes a `StandardScaler` to standardize all the features and a ridge regression with L2 regularization. We have several features in the model, listed here:
- total_fat
- sugar
- sodium
- protein
- saturated_fat
- carbohydrates
- minutes
- avg_rating

All of these are ordinal.

As said above, all that was used to encode all values was `sklearn`'s `StandardScaler`

In terms of performance, we can look at some metrics, including mean squared error, and the R² score. These tests are all easily done using `sklearn` once again.

```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R² Score: {r2}")
```
MSE: 2324.047870089264 <br>
R² Score: 0.9928809462294772

Based on this, while there's a quite good R² score, the mean squared error runs quite high, so I would not call this model "good", per se.

### Final Model
We have a few ideas for improvement of the model:

1. Square the carbohydrates and sugar columns. These columns should have more of an impact on calories, so adding two separate column where their data is squared gives them more weight.
2. **Use `sklearn`'s `PolynomialFeatures()` on all columns which are under `nutrition`**. This should allow us to fit a curve to the data rather than the default ridge regression relationship. We used the `interactions_only` tag to capture relationships between columns such as `saturated_fat` and `carbohydrates`.
    - the columns where this was applied were`total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates`.
