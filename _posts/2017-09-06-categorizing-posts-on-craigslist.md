---
layout: post
title: "Categorizing posts on Craigslist"
date: 2017-09-06
categories:
  - Data Science
  - Machine Learning
source_url: https://datastronomy.com/categorizing-posts-on-craigslist/
---
<p>The purpose of this project is to build a classifier which categorizes posts from Craigslist into the correct category. This tool is useful for categorizing posts that are in catch-all sections such as the &#8220;bartering&#8221; and &#8220;for sale&#8221; categories.</p>
<p></p>
<div class="cell border-box-sizing text_cell rendered">
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Outline">Outline<a class="anchor-link" href="#Outline">¶</a></h3>
<ol>
<li>Introduction
<ol>
<li>Motivation for building this</li>
<li>Exact goals</li>
</ol>
</li>
<li>The challenges that I went through
<ol>
<li>Rushing to deep learning and TensorFlow</li>
<li>Andrew Ng machine learning course</li>
<li>Using eBay data</li>
</ol>
</li>
<li></li>
</ol>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Import-goop">Import goop<a class="anchor-link" href="#Import-goop">¶</a></h2>
<p>Most of the lines below are imports for functions and libraries that we&#8217;ll be using. We import two libraries for this project from the <code>lib</code> folder in the repository.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [4]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">__future__</span> <span class="k">import</span> <span class="n">print_function</span>

<span class="kn">import</span> <span class="nn">sys</span>
<span class="n">sys</span><span class="o">.</span><span class="n">path</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="s1">'..'</span><span class="p">)</span>

<span class="c1"># Libraries functions that were built for this project</span>
<span class="c1"># or copied and pasted from elsewhere</span>
<span class="kn">from</span> <span class="nn">lib.item_selector</span> <span class="k">import</span> <span class="n">ItemSelector</span>
<span class="kn">from</span> <span class="nn">lib.model_performance_plotter</span> <span class="k">import</span> <span class="n">plot_learning_curve</span>

<span class="kn">import</span> <span class="nn">json</span>
<span class="kn">import</span> <span class="nn">pandas</span>
<span class="kn">from</span> <span class="nn">pprint</span> <span class="k">import</span> <span class="n">pprint</span>
<span class="kn">from</span> <span class="nn">sklearn.base</span> <span class="k">import</span> <span class="n">BaseEstimator</span>
<span class="kn">from</span> <span class="nn">sklearn.externals</span> <span class="k">import</span> <span class="n">joblib</span>
<span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">CountVectorizer</span>
<span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">TfidfTransformer</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">GridSearchCV</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.pipeline</span> <span class="k">import</span> <span class="n">FeatureUnion</span>
<span class="kn">from</span> <span class="nn">sklearn.pipeline</span> <span class="k">import</span> <span class="n">Pipeline</span>
<span class="kn">from</span> <span class="nn">time</span> <span class="k">import</span> <span class="n">time</span>

<span class="sd">"""File to load category mapping from"""</span>
<span class="n">CATEGORY_FILE</span> <span class="o">=</span> <span class="s1">'data/categories.json'</span>
<span class="sd">"""File to load data set from"""</span>
<span class="n">DATA_FILE</span> <span class="o">=</span> <span class="s1">'data/cl_posts.csv'</span>
<span class="sd">"""File to save the complete model into"""</span>
<span class="n">MODEL_FILE</span> <span class="o">=</span> <span class="s1">'out/cl_model.pkl'</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Load-and-explore-the-data">Load and explore the data<a class="anchor-link" href="#Load-and-explore-the-data">¶</a></h2>
<p>Use <em>pandas</em> to load Craigslist posts from a CSV file, then drop all examples within that have any <code>N/A</code> or <code>NaN</code> fields.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [6]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">data</span> <span class="o">=</span> <span class="n">pandas</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="n">DATA_FILE</span><span class="p">)</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">dropna</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [7]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1"># Display the first few examples of the data set</span>
<span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[7]:</div>
<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {<br />        text-align: right;<br />    }</p>
<p>    .dataframe thead th {<br />        text-align: left;<br />    }</p>
<p>    .dataframe tbody tr th {<br />        vertical-align: top;<br />    }<br /></style>
<table class="dataframe" border="1">
<thead>
<tr style="text-align: right;">
<th></th>
<th>title</th>
<th>description</th>
<th>category</th>
<th>url</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Are You a Married Woman Looking for Two Guys?</td>
<td>We&#8217;re two fun, discreet married white professi&#8230;</td>
<td>men seeking women</td>
<td>https://chicago.craigslist.org/chc/m4w/d/are-y&#8230;</td>
</tr>
<tr>
<th>1</th>
<td>Producing Consultant</td>
<td>Building effective and collaborative relations&#8230;</td>
<td>business/mgmt</td>
<td>https://iowacity.craigslist.org/bus/d/producin&#8230;</td>
</tr>
<tr>
<th>2</th>
<td>Need ride to tampa thursday for court!!</td>
<td>I am a single mother fighting for custody of m&#8230;</td>
<td>rideshare</td>
<td>https://ocala.craigslist.org/rid/d/need-ride-t&#8230;</td>
</tr>
<tr>
<th>3</th>
<td>Corsair GS 800 Desktop Power Supply</td>
<td>Selling my Corsair GS 800 Desktop Power Supply&#8230;</td>
<td>computer parts &#8211; by owner</td>
<td>https://blacksburg.craigslist.org/sop/d/corsai&#8230;</td>
</tr>
<tr>
<th>4</th>
<td>Free MCAT Quiz for premed students: Can you th&#8230;</td>
<td>Free MCAT Quiz for premed students: Can you th&#8230;</td>
<td>lessons &amp; tutoring</td>
<td>https://albuquerque.craigslist.org/lss/d/free-&#8230;</td>
</tr>
<tr>
<th>5</th>
<td>Wanted Classic Cars and Trucks Any Condition..</td>
<td>Call/text 1.765.613.313one Price Pending Condi&#8230;</td>
<td>wanted &#8211; by owner</td>
<td>https://richmondin.craigslist.org/wan/d/wanted&#8230;</td>
</tr>
<tr>
<th>6</th>
<td>Massage Therapist Wanted</td>
<td>Ontario Family Chiropractic is a holistic base&#8230;</td>
<td>healthcare</td>
<td>https://rochester.craigslist.org/hea/d/massage&#8230;</td>
</tr>
<tr>
<th>7</th>
<td>Lease Take Over at Manchester Motorworks 1 bed&#8230;</td>
<td>Manchester Motorworks is offering a 1 bedroom &#8230;</td>
<td>sublets &amp; temporary</td>
<td>https://richmond.craigslist.org/sub/d/lease-ta&#8230;</td>
</tr>
<tr>
<th>8</th>
<td>🚗 DENVER CAR OWNERS: PAY FOR YOUR CAR BY RENTI&#8230;</td>
<td>Turo is a peer-to-peer car sharing marketplace&#8230;</td>
<td>et cetera</td>
<td>https://denver.craigslist.org/etc/d/denver-car&#8230;</td>
</tr>
<tr>
<th>9</th>
<td>Trunk Mounted Bike Rack w/ 3 Spaces &#8211; Universa&#8230;</td>
<td>Trunk Mounted Bike Rack w/ 3 Spaces &#8211; Universa&#8230;</td>
<td>bicycle parts &#8211; by owner</td>
<td>https://cosprings.craigslist.org/bop/d/trunk-m&#8230;</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [8]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1"># Display the fields of the data set</span>
<span class="nb">list</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[8]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>['title', 'description', 'category', 'url']</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Map-Craigslist-categories-to-our-application-categories">Map Craigslist categories to our application categories<a class="anchor-link" href="#Map-Craigslist-categories-to-our-application-categories">¶</a></h2>
<p>The categories that Craigslist</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="sd">"""</span>
<span class="sd">Load category map to convert from Craigslist categories to our own</span>
<span class="sd">local app categories.</span>
<span class="sd">"""</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">CATEGORY_FILE</span><span class="p">)</span> <span class="k">as</span> <span class="n">handle</span><span class="p">:</span>
    <span class="n">category_map</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">loads</span><span class="p">(</span><span class="n">handle</span><span class="o">.</span><span class="n">read</span><span class="p">())</span>

<span class="sd">"""Load example data using Pandas"""</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="c1"># data, _ = train_test_split(data, test_size=0.5)</span>

<span class="sd">"""Remove all examples with null fields"""</span>


<span class="sd">"""Strip out all "X - by owner", etc. text."""</span>
<span class="n">data</span><span class="p">[</span><span class="s1">'category'</span><span class="p">],</span> <span class="n">_</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="s1">'category'</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">' -'</span><span class="p">,</span> <span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">str</span>

<span class="sd">"""Remap all Craigslist categories to the categories for our use case"""</span>
<span class="n">data</span><span class="p">[</span><span class="s1">'category'</span><span class="p">]</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="n">to_replace</span><span class="o">=</span><span class="n">category_map</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="sd">"""</span>
<span class="sd">Drop all examples with null fields again; this time the categories that</span>
<span class="sd">we're skipping.</span>
<span class="sd">"""</span>
<span class="n">data</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">dropna</span><span class="p">()</span>

<span class="nb">print</span><span class="p">(</span><span class="s1">'All categories:</span><span class="se">\n</span><span class="s1">'</span><span class="p">,</span> <span class="n">data</span><span class="o">.</span><span class="n">category</span><span class="o">.</span><span class="n">value_counts</span><span class="p">())</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Training-and-test-data-split">Training and test data split<a class="anchor-link" href="#Training-and-test-data-split">¶</a></h2>
<p>GridSearchCV already splits a cross validation data set from the training set.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [18]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">train</span><span class="p">,</span> <span class="n">test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.1</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Data-pipeline">Data pipeline<a class="anchor-link" href="#Data-pipeline">¶</a></h2>
<p>Pipeline the process to make it more clear what&#8217;s going on, use less<br />
memory, and enable faster insertion of new steps.</p>
<h3 id="FeatureUnion">FeatureUnion<a class="anchor-link" href="#FeatureUnion">¶</a></h3>
<p>A FeatureUnion allows for unifying multiple input features so that<br />
the model trains itself on all of them.</p>
<h3 id="selector">selector<a class="anchor-link" href="#selector">¶</a></h3>
<p>Select this column only for the purposes of this step of the<br />
pipeline.</p>
<p>Example:</p>
<div class="highlight">
<pre><span class="p">{</span>
    <span class="err">'title':</span> <span class="err">'Lagavulin</span> <span class="err">16',</span>
    <span class="err">'description':</span> <span class="err">'A</span> <span class="err">fine</span> <span class="err">bottle</span> <span class="err">this</span> <span class="err">is.',</span>
    <span class="err">'category':</span> <span class="err">'Alcohol</span> <span class="err">&amp;</span> <span class="err">Spirits'</span>
<span class="p">}</span>
</pre>
</div>
<p>=&gt; <code>'Lagavulin 16'</code></p>
<h3 id="vect">vect<a class="anchor-link" href="#vect">¶</a></h3>
<p>Embed the words in text using a matrix of token counts.</p>
<p>Example:</p>
<div class="highlight">
<pre><span class="p">[</span><span class="s2">"dog cat fish"</span><span class="p">,</span> <span class="s2">"dog cat"</span><span class="p">,</span> <span class="s2">"fish bird"</span><span class="p">,</span> <span class="s2">"bird"</span><span class="p">]</span>
</pre>
</div>
<p>=&gt;</p>
<div class="highlight">
<pre><span class="p">[[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>
 <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>
 <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>
 <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">]]</span>
</pre>
</div>
<h3 id="tfidf">tfidf<a class="anchor-link" href="#tfidf">¶</a></h3>
<p>Deprioritize words that appear very often, such as &#8220;the&#8221;, &#8220;an&#8221;, &#8220;craigslist&#8221;, etc.</p>
<p>Example:</p>
<div class="highlight">
<pre><span class="p">[[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>
 <span class="p">[</span><span class="mi">2</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>
 <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]]</span>
</pre>
</div>
<p>=&gt;</p>
<div class="highlight">
<pre><span class="p">[[</span> <span class="mf">0.81940995</span><span class="p">,</span>  <span class="mi">0</span><span class="err">.</span>        <span class="p">,</span>  <span class="mf">0.57320793</span><span class="p">],</span>
 <span class="p">[</span> <span class="mi">1</span><span class="err">.</span>        <span class="p">,</span>  <span class="mi">0</span><span class="err">.</span>        <span class="p">,</span>  <span class="mi">0</span><span class="err">.</span>        <span class="p">],</span>
 <span class="p">[</span> <span class="mi">1</span><span class="err">.</span>        <span class="p">,</span>  <span class="mi">0</span><span class="err">.</span>        <span class="p">,</span>  <span class="mi">0</span><span class="err">.</span>        <span class="p">]]</span>
</pre>
</div>
<h3 id="clf">clf<a class="anchor-link" href="#clf">¶</a></h3>
<p><code>clf</code> is the classifier that we feed the data from the data pipeline into. In this case, we choose <code>LogisticRegression</code> since it&#8217;s one of the known best ones for text classification. The others are 1) <code>LinearSVC</code>, which is effectively just a linear regression and is similar to <code>LogisticRegression</code>, and 2) neural nets, which without very complicated convolutional and recurrent networks don&#8217;t give us much of an advantage over more classic methods.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [19]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">pipeline</span> <span class="o">=</span> <span class="n">Pipeline</span><span class="p">([</span>
    <span class="p">(</span><span class="s1">'union'</span><span class="p">,</span> <span class="n">FeatureUnion</span><span class="p">(</span>
        <span class="n">transformer_list</span><span class="o">=</span><span class="p">[</span>
            <span class="p">(</span><span class="s1">'title'</span><span class="p">,</span> <span class="n">Pipeline</span><span class="p">([</span>
                <span class="p">(</span><span class="s1">'selector'</span><span class="p">,</span> <span class="n">ItemSelector</span><span class="p">(</span><span class="n">key</span><span class="o">=</span><span class="s1">'title'</span><span class="p">)),</span>
                <span class="p">(</span><span class="s1">'vect'</span><span class="p">,</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="n">stop_words</span><span class="o">=</span><span class="s1">'english'</span><span class="p">,</span>
                                         <span class="n">decode_error</span><span class="o">=</span><span class="s1">'replace'</span><span class="p">,</span>
                                         <span class="n">strip_accents</span><span class="o">=</span><span class="s1">'ascii'</span><span class="p">,</span>
                                         <span class="n">max_df</span><span class="o">=</span><span class="mf">0.8</span><span class="p">)),</span>
                <span class="p">(</span><span class="s1">'tfidf'</span><span class="p">,</span> <span class="n">TfidfTransformer</span><span class="p">(</span><span class="n">smooth_idf</span><span class="o">=</span><span class="kc">False</span><span class="p">))</span>
            <span class="p">])),</span>
            <span class="p">(</span><span class="s1">'description'</span><span class="p">,</span> <span class="n">Pipeline</span><span class="p">([</span>
                <span class="p">(</span><span class="s1">'selector'</span><span class="p">,</span> <span class="n">ItemSelector</span><span class="p">(</span><span class="n">key</span><span class="o">=</span><span class="s1">'description'</span><span class="p">)),</span>
                <span class="p">(</span><span class="s1">'vect'</span><span class="p">,</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="n">stop_words</span><span class="o">=</span><span class="s1">'english'</span><span class="p">,</span>
                                         <span class="n">decode_error</span><span class="o">=</span><span class="s1">'replace'</span><span class="p">,</span>
                                         <span class="n">strip_accents</span><span class="o">=</span><span class="s1">'ascii'</span><span class="p">,</span>
                                         <span class="n">binary</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
                                         <span class="n">max_df</span><span class="o">=</span><span class="mf">0.8</span><span class="p">,</span>
                                         <span class="n">min_df</span><span class="o">=</span><span class="mi">10</span><span class="p">)),</span>
                <span class="p">(</span><span class="s1">'tfidf'</span><span class="p">,</span> <span class="n">TfidfTransformer</span><span class="p">(</span><span class="n">smooth_idf</span><span class="o">=</span><span class="kc">False</span><span class="p">))</span>
            <span class="p">]))</span>
        <span class="p">]</span>
    <span class="p">)),</span>
    <span class="p">(</span><span class="s1">'clf'</span><span class="p">,</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">dual</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">class_weight</span><span class="o">=</span><span class="s1">'balanced'</span><span class="p">))</span>
<span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Pipeline-parameters">Pipeline parameters<a class="anchor-link" href="#Pipeline-parameters">¶</a></h2>
<p>We can optionally set our pipeline parameters to get more control over each step. In the code above, the optimal parameters are already filled out.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">parameters</span> <span class="o">=</span> <span class="p">{</span>
    <span class="c1"># Controls on regression model.</span>
    <span class="c1"># 'clf__C': [0.1, 0.3, 1, 3, 5, 10, 30, 100, 300, 1000]</span>
    <span class="c1"># 'clf__class_weight': [None, 'balanced'],</span>
    <span class="c1"># 'clf__dual': [True, False],</span>

    <span class="c1"># Controls on word vectorization.</span>
    <span class="c1"># 'union__title__vect__max_df': [0.8, 0.85, 0.9, 0.95, 1],</span>
    <span class="c1"># 'union__title__vect__min_df': [1, 10],</span>
    <span class="c1"># 'union__title__vect__ngram_range': [(1, 1), (1, 2)],</span>
    <span class="c1"># 'union__description__vect__ngram_range': [(1, 1), (1, 2)],</span>
    <span class="c1"># 'union__description__vect__max_df': [0.8, 0.85, 0.9, 0.95, 1],</span>
    <span class="c1"># 'union__description__vect__min_df': [1, 10, 100],</span>

    <span class="c1"># Controls on TfIdf normalization.</span>
    <span class="c1"># 'union__title__tfidf__norm': [None, 'l1', 'l2'],</span>
    <span class="c1"># 'union__title__tfidf__use_idf': [True, False],</span>
    <span class="c1"># 'union__title__tfidf__smooth_idf': [True, False],</span>
    <span class="c1"># 'union__title__tfidf__sublinear_tf': [False, True],</span>
    <span class="c1"># 'union__description__tfidf__norm': [None, 'l1', 'l2'],</span>
    <span class="c1"># 'union__description__tfidf__use_idf': [True, False],</span>
    <span class="c1"># 'union__description__tfidf__smooth_idf': [True, False],</span>
    <span class="c1"># 'union__description__tfidf__sublinear_tf': [False, True],</span>
<span class="p">}</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Train-the-model">Train the model<a class="anchor-link" href="#Train-the-model">¶</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">grid_search</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">pipeline</span><span class="p">,</span> <span class="n">parameters</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s1">'Performing grid search...'</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'Pipeline: '</span><span class="p">,</span> <span class="p">[</span><span class="n">name</span> <span class="k">for</span> <span class="n">name</span><span class="p">,</span> <span class="n">__</span> <span class="ow">in</span> <span class="n">pipeline</span><span class="o">.</span><span class="n">steps</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">'Parameters: '</span><span class="p">)</span>
<span class="n">pprint</span><span class="p">(</span><span class="n">parameters</span><span class="p">)</span>
<span class="n">t0</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span>
<span class="n">grid_search</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train</span><span class="p">[[</span><span class="s1">'title'</span><span class="p">,</span> <span class="s1">'description'</span><span class="p">]],</span> <span class="n">train</span><span class="p">[</span><span class="s1">'category'</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Done in </span><span class="si">%0.3f</span><span class="s2">s"</span> <span class="o">%</span> <span class="p">(</span><span class="n">time</span><span class="p">()</span> <span class="o">-</span> <span class="n">t0</span><span class="p">))</span>
<span class="nb">print</span><span class="p">()</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"Best score: </span><span class="si">%0.3f</span><span class="s2">"</span> <span class="o">%</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">best_score_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Best parameters set:"</span><span class="p">)</span>
<span class="n">best_parameters</span> <span class="o">=</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">best_estimator_</span><span class="o">.</span><span class="n">get_params</span><span class="p">()</span>
<span class="k">for</span> <span class="n">param_name</span> <span class="ow">in</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">parameters</span><span class="o">.</span><span class="n">keys</span><span class="p">()):</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\t</span><span class="si">%s</span><span class="s2">: </span><span class="si">%r</span><span class="s2">"</span> <span class="o">%</span> <span class="p">(</span><span class="n">param_name</span><span class="p">,</span> <span class="n">best_parameters</span><span class="p">[</span><span class="n">param_name</span><span class="p">]))</span>

<span class="n">score</span> <span class="o">=</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">test</span><span class="p">[[</span><span class="s1">'title'</span><span class="p">,</span> <span class="s1">'description'</span><span class="p">]],</span> <span class="n">test</span><span class="p">[</span><span class="s1">'category'</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Test accuracy: </span><span class="si">%f</span><span class="s2">"</span> <span class="o">%</span> <span class="n">score</span><span class="p">)</span>

<span class="n">joblib</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="n">grid_search</span><span class="p">,</span> <span class="n">MODEL_FILE</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Performing grid search...
Pipeline:  ['union', 'clf']
Parameters:
{}
Fitting 3 folds for each of 1 candidates, totalling 3 fits
[CV]  ................................................................
[CV]  ................................................................
[CV]  ................................................................
[CV] ....................... , score=0.7874693401652609, total=33.0min
[CV] ....................... , score=0.7887705122605603, total=33.1min
[CV] ....................... , score=0.7877126558241279, total=34.9min
</pre>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>[Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed: 36.5min remaining:    0.0s
[Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed: 36.5min finished
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Plot-the-data">Plot the data<a class="anchor-link" href="#Plot-the-data">¶</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [ ]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">plot_learning_curve</span><span class="p">(</span><span class="n">grid_search</span><span class="o">.</span><span class="n">best_estimator_</span><span class="p">,</span>
                    <span class="s1">'Item Categorizer'</span><span class="p">,</span>
                    <span class="n">train</span><span class="p">[[</span><span class="s1">'title'</span><span class="p">,</span> <span class="s1">'description'</span><span class="p">]],</span>
                    <span class="n">train</span><span class="p">[</span><span class="s1">'category'</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
</div>

