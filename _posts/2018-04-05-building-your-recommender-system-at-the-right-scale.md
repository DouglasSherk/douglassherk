---
layout: post
title: "Building your recommender system at the right scale"
date: 2018-04-05
categories:
  - Data Infrastructure
  - Recommender Systems
source_url: https://datastronomy.com/building-your-recommender-system-at-the-right-scale/
---
<div style="display: none;">
<p>OutlineOpening &#8220;everyone wants to scale&#8221;</p>
<p>Recommender systems are just like any other data system<br />
&#8211; Keep it as simple as possible<br />
&#8211; Avoid undifferentiated heavy lifting</p>
<p>The problems recommender systems face at each level of scale<br />
&#8211; Toy problems<br />
&#8211; Medium-size problems<br />
&#8211; Out-of-core training<br />
&#8211; Real-time recommendations<br />
&#8211; Storing the recommendations<br />
&#8211; Large scale<br />
&#8211; Apache Spark, OLS<br />
&#8211; Setting up a cluster<br />
&#8211; Massive scale<br />
&#8211; Still use off-the-shelf models<br />
&#8211; Mostly about feature engineering</p>
<p>How to decide on where you fit<br />
&#8211; About</p>
</div>
<div style="display: none;">
<p><figure id="attachment_235" aria-describedby="caption-attachment-235" style="width: 290px" class="wp-caption aligncenter"><img data-recalc-dims="1" loading="lazy" decoding="async" class="size-full wp-image-235" src="https://i0.wp.com/www.datastronomy.com/wp-content/uploads/2018/04/dog-firehose.jpg?resize=290%2C174" alt="A dog drinking from a hose" width="290" height="174" /><figcaption id="caption-attachment-235" class="wp-caption-text">An MVP lapping up the incoming data from the 10K concurrents on launch day.</figcaption></figure></p>
<p>An engineer is scoping out a data system design and the first thing that comes to mind is how it&#8217;ll work at scale. He&#8217;s just starting on the prototype, so the product has no users yet, but he wants to make sure that it can eventually accommodate thousands of concurrents! With the architecture divorced from reality, inevitably the end result is late or it doesn&#8217;t solve the problem.</p>
<p>Meanwhile, in practice he could comfortably build the application with the LAMP stack on a micro AWS instance sans database indexes.</p>
<p>I see such mismanagement happen too often. As a committed long-term planner, I also feel the urge to think ahead. Unless you&#8217;re managing budgets of multiple millions of dollars, there&#8217;s rarely business value in working ahead on problems that will manifest later than two to three months from now.</p>
</div>
<h1>Scoping recommender systems</h1>
<p>Recommender systems are just like other data systems. When building recommenders, you should be asking yourself &#8220;how can I make this happen with as little complexity as possible?&#8221; There&#8217;s a wealth of information available on when and how to deploy database shards, caches, proxies, and other scaling tools. But there&#8217;s a comparative dearth of information on what problems you&#8217;ll face when building recommender systems.</p>
<p>Throughout this article, I&#8217;m going to assume that you&#8217;re building a collaborative filtering (CF) system, but many of the same challenges apply to content-based filtering systems too.</p>
<p>Having first-hand experience—or failing that, second-hand experience—is the key to hitting the sweet spot between simplicity and complexity. There are five clear scales of recommender systems.</p>
<h2>Toy scale</h2>
<p>Ah, the toy-scale recommender systems. The internet is chalk full of them, so if this is the point you&#8217;re at, then you&#8217;re in luck. Toy problems are characterized by offline training and prediction, no live deployment, and datasets that fit in memory even when represented by dense matrices.</p>
<p>There&#8217;s no shame in building a toy-scale system. They&#8217;re fun to develop because they require minimal engineering work and get you close to the underlying algorithms. <a href="http://surpriselib.com/">Surprise</a> lets you go from nothing to <a href="http://surprise.readthedocs.io/en/stable/getting_started.html#train-test-split-py">computing movie recommendations in twenty lines of code</a>!</p>
<p>Even if a system is live in production, I would still lump it into this category if it has fewer than 10K users. All you have to do to stand it up is expose a REST API by wrapping the model with a lightweight HTTP server.</p>
<h3>Evaluating tools</h3>
<p>When you&#8217;re in this category and you&#8217;re evaluating tools, you&#8217;ll want them to be as easy to set up and understand as possible. Surprise, mentioned above, is an excellent choice.</p>
<h2>Small scale</h2>
<p>A system goes from toy to small scale when it&#8217;s deployed live in production with ~10K users or more. Typical problems at this scale are annoyances rather than true challenges.</p>
<h3>CF algorithm performance</h3>
<p>Complexity analysis reveals that some CF algorithms will break down with more than a few thousand users. One example is <a href="http://scikit-learn.org/stable/modules/neighbors.html">k-nearest neighbors</a> (kNN) for which the complexity is described by <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/building-your-recommender-system-at-the-right-scale/quicklatex.com-306ba3d974b5cbd110589f3aa7b0a710_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#79;&#40;&#124;&#117;&#124;&#32;&#92;&#99;&#100;&#111;&#116;&#32;&#124;&#102;&#124;&#41;" title="Rendered by QuickLaTeX.com" height="19" width="80" style="vertical-align: -5px;"/>, where <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/building-your-recommender-system-at-the-right-scale/quicklatex.com-75d280d237d1a017ea0e0d3eec38c063_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#92;&#108;&#118;&#101;&#114;&#116;&#32;&#117;&#32;&#92;&#114;&#118;&#101;&#114;&#116;" title="Rendered by QuickLaTeX.com" height="19" width="16" style="vertical-align: -5px;"/> is the number of users, and <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/building-your-recommender-system-at-the-right-scale/quicklatex.com-c262dff1b34e7cb0c1938df8452b6817_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#124;&#102;&#124;" title="Rendered by QuickLaTeX.com" height="19" width="17" style="vertical-align: -5px;"/> is the number of features. You can optimize your code using <a href="http://docs.cython.org/en/latest/src/quickstart/overview.html">Cython</a> or a JVM language, or you can use a kNN optimization like <a href="http://scikit-learn.org/stable/modules/neighbors.html#ball-tree">ball trees</a>, but you&#8217;re usually better served by switching to an <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/building-your-recommender-system-at-the-right-scale/quicklatex.com-b992bb94505bc794663eeca11235336f_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#79;&#40;&#124;&#102;&#124;&#41;" title="Rendered by QuickLaTeX.com" height="19" width="47" style="vertical-align: -5px;"/> algorithm such as <a href="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html">SVD</a>.</p>
<h3>Memory hogging</h3>
<p>You may also notice that your CF algorithm is consuming lots of memory (2 GB or more). A back-of-the-napkin calculation would show that storing the ratings in a dense matrix would consume <em>10,000 users × 10,000 features × 1 byte per rating</em> = only <em>100 MB.</em> However, there are inefficiencies in moving the data around and in the internals of some CF algorithms. Chances are that you won&#8217;t be able to ignore this problem, so you&#8217;ll have to fix it by making sure that you&#8217;re never copying the dataset, vertically scaling the training machine, or switching to <a href="https://en.wikipedia.org/wiki/Sparse_matrix#Storing_a_sparse_matrix">compressed sparse matrices</a>.</p>
<h3>Evaluating tools</h3>
<p>Surprise will still serve you well at small scale, but you&#8217;ll have to be mindful of memory use and of which CF algorithm you&#8217;re using.</p>
<h2>Medium scale</h2>
<p>What I call medium scale is when the challenges start getting real. As a rule of thumb, you&#8217;ll have moved to this scale with 100K+ users and 10K+ features. Several assumptions that we could hand-wave away at the toy and small scales fall apart at this point.</p>
<h3>Compressed sparse matrices</h3>
<p>Main article: <a href="http://www.datastronomy.com/understanding-sparse-matrices-for-recommender-systems/">Understanding sparse matrices for recommender systems</a></p>
<p>The problems fitting the training dataset in memory will only get worse as the number of users and features grow. In the section for small scale, I alluded to switching to compressed sparse matrices. As you transition to medium scale, you will have no choice.</p>
<p>Let&#8217;s see why. Assume that you have 100K users and 10K features. To store the ratings in a dense matrix, you would need <em>100,000 users × 10,000 features × 1 byte per rating</em> = <em>1 GB.</em> That figure doesn&#8217;t seem like much, but watch what happens when the number of users and number of features both double: <em>200,000 users × 20,000 features × 1 byte per rating</em> = 4<em> GB</em>!</p>
<p>These examples show that the space complexity of matrix factorization algorithms climbs in <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/building-your-recommender-system-at-the-right-scale/quicklatex.com-306ba3d974b5cbd110589f3aa7b0a710_l3-2.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#79;&#40;&#124;&#117;&#124;&#32;&#92;&#99;&#100;&#111;&#116;&#32;&#124;&#102;&#124;&#41;" title="Rendered by QuickLaTeX.com" height="19" width="80" style="vertical-align: -5px;"/>. A startup that gets to medium scale will probably be growing at 20% month-over-month, so the training dataset would consume 16 GB of RAM after four months. Sure, you can limit the number of users and features in the training set, but then you&#8217;re just buying time to avoid the inevitable.</p>
<p>Fortunately, using compressed sparse matrix representations can solve this problem. This technique takes advantage of the typical 1-2% density in rating data by compressing the ratings into a format that avoids storing missing ratings. Let&#8217;s redo the last calculation with a sparse matrix and assume a rating density of 1%: <em>200,000 users × 20,000 features × 3 bytes per rating × 1% density</em> = a far more manageable <em>120 MB</em>.</p>
<h3>Evaluating tools</h3>
<p>Remember that the output of a matrix factorization algorithm is always a dense matrix. Surprise will no longer serve you at this scale; you&#8217;ll need more fine-grained control over the validation dataset. You might continue to use this framework as the backend for a service that passes the validation dataset in batches and computes metrics (like RMSE) incrementally instead. When you&#8217;re evaluating tools and solutions at this scale, you&#8217;ll want to start thinking further out than two to three months from now.</p>
<h2>Large scale</h2>
<p>As you may have already noticed, the engineering challenges have been growing exponentially, and will (spoiler alert) continue to do so. Large scale is characterized by 10M users and 100K features. You can redo the calculations above to convince yourself that you&#8217;ll need new techniques for this scale.</p>
<h3>Distributed training</h3>
<p>At large scale, the dataset used for training will no longer fit in memory, even when stored sparsely.</p>
<p>One way to work around this problem is to do training out-of-core, which is when batches of training data are incrementally fed to the model. Surprise and most other recommender system frameworks lack incremental training algorithms, so you have to hand-roll one yourself.</p>
<p><a href="http://scikit-learn.org/stable/index.html">SciKit-Learn</a> has a module called <a href="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html">IncrementalPCA</a>. Unfortunately, this model has two problems. First, it doesn&#8217;t accept sparse data, so you&#8217;ll have to incrementally feed it dense data. Doing so is ludicrously slow, especially with this number of features and users. Second, the model doesn&#8217;t make predictions that are anywhere near as accurate as Surprise or SciKit-Learn&#8217;s <a href="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html">TruncatedSVD</a>. This inaccuracy is probably due to the data sparsity rather than an inherent problem with IncrementalPCA.</p>
<p>The better solution is to use Apache Spark and its built-in MLLib toolkit. Jose A Dianes wrote an excellent blog post on how to get started using <a href="https://www.codementor.io/jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw">Spark to make movie recommendations</a>, but be mindful that this approach comes with a whole new set of problems.</p>
<h2>Big data scale</h2>
<p>Only large companies with teams dedicated to information retrieval encounter big data-scale problems. Even with the terabytes or more of data that these firms process, the bulk of the challenges come from coordinating teams, data pipelining, feature engineering, and vectorization. Most <a href="https://www.coursera.org/learn/matrix-factorization/lecture/KnGh1/industry-practical-issues-inteview-with-anmol-bhasin">use off-the-shelf models</a> to make recommendations and employ far fewer data scientists than engineers.</p>
<p>As you probably guessed, this article is not targeted at people or teams building big data systems. It&#8217;s still interesting to know how the challenges change as the business evolves.</p>
<h1>Wrapping it up</h1>
<p><strong>Figure out where you are and build for that scale</strong>. Chances are that your problem can be solved with a small or medium-scale recommender system. Focusing on building only what&#8217;s necessary is the way to get work done quickly and effectively!</p>

