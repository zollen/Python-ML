https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution


The short version is that the Beta distribution can be understood as representing a distribution 
of probabilities, that is, it represents all the possible values of a probability when we don't 
know what that probability is. Here is my favorite intuitive explanation of this:


Anyone who follows baseball is familiar with batting averages—simply the number of times a player 
gets a base hit divided by the number of times he goes up at bat (so it's just a percentage between 
0 and 1). .266 is in general considered an average batting average, while .300 is considered an 
excellent one.

Imagine we have a baseball player, and we want to predict what his season-long batting average 
will be. You might say we can just use his batting average so far- but this will be a very poor 
measure at the start of a season! If a player goes up to bat once and gets a single, his batting 
average is briefly 1.000, while if he strikes out, his batting average is 0.000. It doesn't get 
much better if you go up to bat five or six times- you could get a lucky streak and get an average 
of 1.000, or an unlucky streak and get an average of 0, neither of which are a remotely good 
predictor of how you will bat that season.

Why is your batting average in the first few hits not a good predictor of your eventual batting 
average? When a player's first at-bat is a strikeout, why does no one predict that he'll never 
get a hit all season? Because we're going in with prior expectations. We know that in history, 
most batting averages over a season have hovered between something like .215 and .360, with some 
extremely rare exceptions on either side. We know that if a player gets a few strikeouts in a row 
at the start, that might indicate he'll end up a bit worse than average, but we know he probably 
won't deviate from that range.

Given our batting average problem, which can be represented with a binomial distribution (a series 
of successes and failures), the best way to represent these prior expectations (what we in 
statistics just call a prior) is with the Beta distribution- it's saying, before we've seen the 
player take his first swing, what we roughly expect his batting average to be. The domain of the 
Beta distribution is (0, 1), just like a probability, so we already know we're on the right 
track, but the appropriateness of the Beta for this task goes far beyond that.

We expect that the player's season-long batting average will be most likely around .27, but that 
it could reasonably range from .21 to .35. This can be represented with a Beta distribution with 
parameters α=81 and β=219:


I came up with these parameters for two reasons:

The mean is αα+β=8181+219=.270
As you can see in the plot, this distribution lies almost entirely within (.2, .35)- the 
reasonable range for a batting average.