# NPO-Prediction-Application

The application is packaged into a single executable file. Users need to double click the executable and wait for the application to open. Once it opens, there’s a textbox at the top and three labeled buttons. Users simply need to enter a year or range of years into the textbox, and click on the function they want.

The first two functions, the demand predictor and the heatmap generator will ask for a database. As mentioned previously, the demand predictor fits equations to the data, and therefore will produce better predictions as the years go on, providing an system that will grow alongside your organization. The heatmap generator will produce two heatmaps, one split up by counties and another split up by zipcodes. Zipshapes files are needed to run these functions.

The third functionality is a demographics based zipcode classifier. Users simply need to input a list of zipcodes and the algorithm will output whether or not these zipcode will have more than 25 items of demand. This algorithm should be useful for expansion into new areas, such as Dallas, since donations records do not yet exist. 
