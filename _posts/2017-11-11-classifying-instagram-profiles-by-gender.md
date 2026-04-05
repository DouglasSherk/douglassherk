---
layout: post
title: "Classifying Instagram profiles by gender"
date: 2017-11-11
categories:
  - Data Science
  - Machine Learning
source_url: https://datastronomy.com/classifying-instagram-profiles-by-gender/
---
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The purpose of this project is to <strong>create a model which, given an Instagram user&#8217;s profile, predicts their gender as accurately as possible</strong>. The motivation for this undertaking is to be able to target for marketing purposes Instagram users of specific demographics. The model is trained using labeled text-based profile data passed through a tuned logistic regression model. The model parameters are optimized using the AUROC metric to reduce variability in the precision and recall of predictions for each gender. The resulting model achieves 90% overall accuracy from a dataset of 20,000, though it deviates substantially in the recall of each gender.</p>
</div>
</div>
</div>
<p></p>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Introduction">Introduction<a class="anchor-link" href="#Introduction">¶</a></h2>
<p>All supporting files for this project can be found in its <a href="https://github.com/DouglasSherk?tab=repositories">GitHub repository</a>.</p>
<p>This project write-up assumes that the reader has a basic understanding of machine learning and statistics concepts including logistic regression, word encodings including bag-of-words, and the terminology surrounding false/true positives/negatives.</p>
<p>The following high-level details are of note:</p>
<ul>
<li><strong>Instagram profiles are mostly text</strong>. The modeling methods that typically perform best for text classification are regressions and neural nets. This project employs logistic regression as its model of choice.</li>
<li><strong>The model is designed to perform equally well on both genders</strong>. This stipulation was a business constraint of which the most significant consequence was the replacement of the cross-validation optimization metric of accuracy with AUROC.</li>
<li><strong>The data pipeline makes heavy use of n-grams and both word and character encodings</strong>. The use of these more complicated bag-of-words encodings improves results substantially beyond simpler 1-gram word-based encodings.</li>
</ul>
<p>Due to the difficulty of obtaining reliable data about genders other than male and female, and the lack of marketing value in these smaller demographics, the following analysis eschews these additional labels. Rest assured, this omission is for economic as opposed to political or social reasons.</p>
<p>For reasons including logistical difficulty and the data constituting business trade secrets, the labeled profiles cannot be posted publicly. One way in which the results of this project can be replicated is by querying the <a href="https://www.instagram.com/developer/endpoints/users">Instagram User API</a> and then labeling the data using <a href="https://www.mturk.com">Amazon Mechanical Turk</a>.</p>
</div>
</div>
</div>
<p></p>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Data-Engineering-and-Cleaning">Data Engineering and Cleaning<a class="anchor-link" href="#Data-Engineering-and-Cleaning">¶</a></h2>
<p>The code in the following section loads, organizes, and formats the labeled training data in such a way that it can later be passed into an off-the-shelf model.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [39]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="sd">"""</span>
<span class="sd">Jupyter notebook boilerplate setup code.</span>
<span class="sd">"""</span>

<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="o">%</span><span class="k">load_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2
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
<pre>The autoreload extension is already loaded. To reload it, use:
  %reload_ext autoreload
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [40]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">import</span> <span class="nn">json</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

<span class="sd">"""</span>
<span class="sd">Files from which to load datasets and labels. In this example, the labels</span>
<span class="sd">are separate from the rest of the user profile data, and these data are</span>
<span class="sd">related using a dictionary.</span>
<span class="sd">"""</span>
<span class="n">DATA_FILES</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'doug_labeled_user_batch.json'</span><span class="p">:</span> <span class="s1">'doug_labels.json'</span><span class="p">,</span>
    <span class="s1">'doug_finaly_labeled_cleaned_batch_2.json'</span><span class="p">:</span> <span class="s1">'doug_labels_batch_2.json'</span>
<span class="p">}</span>

<span class="sd">"""</span>
<span class="sd">Load example data using Pandas.</span>
<span class="sd">"""</span>
<span class="n">datasets</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">profiles_file</span><span class="p">,</span> <span class="n">labels_file</span> <span class="ow">in</span> <span class="n">DATA_FILES</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
    <span class="n">datasets</span><span class="o">.</span><span class="n">append</span><span class="p">({</span>
        <span class="s1">'profiles'</span><span class="p">:</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_json</span><span class="p">(</span><span class="n">profiles_file</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s1">'ISO-8859-1'</span><span class="p">),</span>
        <span class="s1">'labels'</span><span class="p">:</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_json</span><span class="p">(</span><span class="n">labels_file</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s1">'ISO-8859-1'</span><span class="p">)</span>
    <span class="p">})</span>

<span class="s2">"Loaded </span><span class="si">%d</span><span class="s2"> datasets"</span> <span class="o">%</span> <span class="nb">len</span><span class="p">(</span><span class="n">DATA_FILES</span><span class="o">.</span><span class="n">items</span><span class="p">())</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[40]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>'Loaded 2 datasets'</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Check-the-loaded-data">Check the loaded data<a class="anchor-link" href="#Check-the-loaded-data">¶</a></h3>
<p>It&#8217;s best to examine the loaded data to verify that it is in the expected format.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [41]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">total_examples</span> <span class="o">=</span> <span class="mi">0</span>
<span class="k">for</span> <span class="n">dataset</span> <span class="ow">in</span> <span class="n">datasets</span><span class="p">:</span>
    <span class="n">total_examples</span> <span class="o">+=</span> <span class="nb">len</span><span class="p">(</span><span class="n">dataset</span><span class="p">[</span><span class="s1">'profiles'</span><span class="p">])</span>
<span class="s2">"</span><span class="si">%d</span><span class="s2"> total examples"</span> <span class="o">%</span> <span class="n">total_examples</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[41]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>'20460 total examples'</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [42]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">datasets</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s1">'profiles'</span><span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[42]:</div>
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
<th>biography</th>
<th>blocked_by_viewer</th>
<th>connected_fb_page</th>
<th>country_block</th>
<th>external_url</th>
<th>external_url_linkshimmed</th>
<th>followed_by</th>
<th>followed_by_viewer</th>
<th>follows</th>
<th>follows_viewer</th>
<th>&#8230;</th>
<th>has_requested_viewer</th>
<th>id</th>
<th>is_private</th>
<th>is_verified</th>
<th>media</th>
<th>profile_pic_url</th>
<th>profile_pic_url_hd</th>
<th>requested_by_viewer</th>
<th>saved_media</th>
<th>username</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Just me. 19\nsnapchat: abbipandi</td>
<td>False</td>
<td>NaN</td>
<td>False</td>
<td>http://twitter.com/abiigaildg?s=09</td>
<td>http://l.instagram.com/?u=http%3A%2F%2Ftwitter&#8230;</td>
<td>{&#8216;count&#8217;: 256}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 642}</td>
<td>False</td>
<td>&#8230;</td>
<td>False</td>
<td>255372732</td>
<td>True</td>
<td>False</td>
<td>{&#8216;count&#8217;: 182, &#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>abigail13d</td>
</tr>
<tr>
<th>1</th>
<td>Just a 23 year old living in Milwaukee</td>
<td>False</td>
<td>NaN</td>
<td>False</td>
<td>None</td>
<td>None</td>
<td>{&#8216;count&#8217;: 169}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 223}</td>
<td>False</td>
<td>&#8230;</td>
<td>False</td>
<td>899493065</td>
<td>True</td>
<td>False</td>
<td>{&#8216;count&#8217;: 18, &#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>jkst0329</td>
</tr>
<tr>
<th>2</th>
<td>✖️⚠️Follow @billz433⚠️✖️</td>
<td>False</td>
<td>NaN</td>
<td>False</td>
<td>http://www.thiscrush.com/~nicobonta</td>
<td>http://l.instagram.com/?u=http%3A%2F%2Fwww.thi&#8230;</td>
<td>{&#8216;count&#8217;: 111}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 52}</td>
<td>False</td>
<td>&#8230;</td>
<td>False</td>
<td>5566432352</td>
<td>False</td>
<td>False</td>
<td>{&#8216;count&#8217;: 3, &#8216;nodes&#8217;: [{&#8216;__typename&#8217;: &#8216;GraphIm&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>nicobonta18</td>
</tr>
<tr>
<th>3</th>
<td>None</td>
<td>False</td>
<td>NaN</td>
<td>False</td>
<td>None</td>
<td>None</td>
<td>{&#8216;count&#8217;: 71}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 511}</td>
<td>False</td>
<td>&#8230;</td>
<td>False</td>
<td>5416238465</td>
<td>False</td>
<td>False</td>
<td>{&#8216;count&#8217;: 3, &#8216;nodes&#8217;: [{&#8216;__typename&#8217;: &#8216;GraphIm&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>sunnykumar7094</td>
</tr>
<tr>
<th>4</th>
<td>Don&#8217;t worry be happy☺\n🎉wish me 23 Feb🎂\nBigge&#8230;</td>
<td>False</td>
<td>NaN</td>
<td>False</td>
<td>None</td>
<td>None</td>
<td>{&#8216;count&#8217;: 116}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 1143}</td>
<td>False</td>
<td>&#8230;</td>
<td>False</td>
<td>5679439304</td>
<td>False</td>
<td>False</td>
<td>{&#8216;count&#8217;: 2, &#8216;nodes&#8217;: [{&#8216;__typename&#8217;: &#8216;GraphIm&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>purbashamalakar</td>
</tr>
</tbody>
</table>
<p>5 rows × 22 columns</p>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [43]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">datasets</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s1">'labels'</span><span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[43]:</div>
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
<th>gender</th>
<th>id</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>female</td>
<td>255372732</td>
</tr>
<tr>
<th>1</th>
<td>female</td>
<td>899493065</td>
</tr>
<tr>
<th>2</th>
<td>male</td>
<td>5566432352</td>
</tr>
<tr>
<th>3</th>
<td>male</td>
<td>5416238465</td>
</tr>
<tr>
<th>4</th>
<td>male</td>
<td>5679439304</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Notice the presence of the <code>id</code> field in both datasets. This field is the one upon which the profile and label data will be merged.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Combine-the-profiles-with-the-labels">Combine the profiles with the labels<a class="anchor-link" href="#Combine-the-profiles-with-the-labels">¶</a></h3>
<p>In this step, the profiles and labels are merged on the <code>id</code> field present in each. The datasets are then concatenated to produce one large set of examples.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [44]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="k">for</span> <span class="n">dataset</span> <span class="ow">in</span> <span class="n">datasets</span><span class="p">:</span>
    <span class="n">dataset</span><span class="p">[</span><span class="s1">'merged'</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">merge</span><span class="p">(</span><span class="n">left</span><span class="o">=</span><span class="n">dataset</span><span class="p">[</span><span class="s1">'profiles'</span><span class="p">],</span>
                                 <span class="n">right</span><span class="o">=</span><span class="n">dataset</span><span class="p">[</span><span class="s1">'labels'</span><span class="p">],</span>
                                 <span class="n">left_on</span><span class="o">=</span><span class="s1">'id'</span><span class="p">,</span>
                                 <span class="n">right_on</span><span class="o">=</span><span class="s1">'id'</span><span class="p">)</span>

<span class="n">data</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">(</span><span class="nb">map</span><span class="p">(</span><span class="k">lambda</span> <span class="n">dataset</span><span class="p">:</span> <span class="n">dataset</span><span class="p">[</span><span class="s1">'merged'</span><span class="p">],</span> <span class="n">datasets</span><span class="p">))</span>
<span class="n">data</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">value</span><span class="o">=</span><span class="s1">''</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[44]:</div>
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
<th>biography</th>
<th>blocked_by_viewer</th>
<th>connected_fb_page</th>
<th>country_block</th>
<th>external_url</th>
<th>external_url_linkshimmed</th>
<th>followed_by</th>
<th>followed_by_viewer</th>
<th>follows</th>
<th>follows_viewer</th>
<th>&#8230;</th>
<th>id</th>
<th>is_private</th>
<th>is_verified</th>
<th>media</th>
<th>profile_pic_url</th>
<th>profile_pic_url_hd</th>
<th>requested_by_viewer</th>
<th>saved_media</th>
<th>username</th>
<th>gender</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Just me. 19\nsnapchat: abbipandi</td>
<td>False</td>
<td></td>
<td>False</td>
<td>http://twitter.com/abiigaildg?s=09</td>
<td>http://l.instagram.com/?u=http%3A%2F%2Ftwitter&#8230;</td>
<td>{&#8216;count&#8217;: 256}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 642}</td>
<td>False</td>
<td>&#8230;</td>
<td>255372732</td>
<td>True</td>
<td>False</td>
<td>{&#8216;count&#8217;: 182, &#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>abigail13d</td>
<td>female</td>
</tr>
<tr>
<th>1</th>
<td>Just a 23 year old living in Milwaukee</td>
<td>False</td>
<td></td>
<td>False</td>
<td></td>
<td></td>
<td>{&#8216;count&#8217;: 169}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 223}</td>
<td>False</td>
<td>&#8230;</td>
<td>899493065</td>
<td>True</td>
<td>False</td>
<td>{&#8216;count&#8217;: 18, &#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>jkst0329</td>
<td>female</td>
</tr>
<tr>
<th>2</th>
<td>✖️⚠️Follow @billz433⚠️✖️</td>
<td>False</td>
<td></td>
<td>False</td>
<td>http://www.thiscrush.com/~nicobonta</td>
<td>http://l.instagram.com/?u=http%3A%2F%2Fwww.thi&#8230;</td>
<td>{&#8216;count&#8217;: 111}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 52}</td>
<td>False</td>
<td>&#8230;</td>
<td>5566432352</td>
<td>False</td>
<td>False</td>
<td>{&#8216;count&#8217;: 3, &#8216;nodes&#8217;: [{&#8216;__typename&#8217;: &#8216;GraphIm&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>nicobonta18</td>
<td>male</td>
</tr>
<tr>
<th>3</th>
<td></td>
<td>False</td>
<td></td>
<td>False</td>
<td></td>
<td></td>
<td>{&#8216;count&#8217;: 71}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 511}</td>
<td>False</td>
<td>&#8230;</td>
<td>5416238465</td>
<td>False</td>
<td>False</td>
<td>{&#8216;count&#8217;: 3, &#8216;nodes&#8217;: [{&#8216;__typename&#8217;: &#8216;GraphIm&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>sunnykumar7094</td>
<td>male</td>
</tr>
<tr>
<th>4</th>
<td>Don&#8217;t worry be happy☺\n🎉wish me 23 Feb🎂\nBigge&#8230;</td>
<td>False</td>
<td></td>
<td>False</td>
<td></td>
<td></td>
<td>{&#8216;count&#8217;: 116}</td>
<td>False</td>
<td>{&#8216;count&#8217;: 1143}</td>
<td>False</td>
<td>&#8230;</td>
<td>5679439304</td>
<td>False</td>
<td>False</td>
<td>{&#8216;count&#8217;: 2, &#8216;nodes&#8217;: [{&#8216;__typename&#8217;: &#8216;GraphIm&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>https://instagram.flju1-1.fna.fbcdn.net/t51.28&#8230;</td>
<td>False</td>
<td>{&#8216;nodes&#8217;: [], &#8216;page_info&#8217;: {&#8216;end_cursor&#8217;: None&#8230;</td>
<td>purbashamalakar</td>
<td>male</td>
</tr>
</tbody>
</table>
<p>5 rows × 23 columns</p>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Format-gender-to-be-compatible-with-AUROC">Format <code>gender</code> to be compatible with AUROC<a class="anchor-link" href="#Format-gender-to-be-compatible-with-AUROC">¶</a></h3>
<p>The encoding of the <code>gender</code> field is as a string with the value &#8220;male&#8221; or &#8220;female&#8221;. The AUROC optimization metric requires a binary (<code>0</code> or <code>1</code>) value. The code below re-encodes <code>gender</code> into a new <code>gender_enc</code> field in which males are encoded with the value <code>0</code> and females with the value <code>1</code>.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [45]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span>
    <span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="mi">0</span> <span class="k">if</span> <span class="n">x</span><span class="p">[</span><span class="s1">'gender'</span><span class="p">]</span> <span class="o">==</span> <span class="s1">'male'</span> <span class="k">else</span> <span class="mi">1</span><span class="p">,</span>
    <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">data</span><span class="p">[[</span><span class="s1">'gender'</span><span class="p">,</span> <span class="s1">'gender_enc'</span><span class="p">]]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[45]:</div>
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
<th>gender</th>
<th>gender_enc</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>1</th>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>2</th>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>male</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Prepare-the-writing_example-field">Prepare the <code>writing_example</code> field<a class="anchor-link" href="#Prepare-the-writing_example-field">¶</a></h3>
<p>One of the fields provided in the dataset is <code>media</code>, which is an array of metadata about each user&#8217;s photos including the captions. The <code>caption</code> field provides a point of leverage because it is the one section of the user&#8217;s profile in which they can write freeform text. This field is also high leverage for the project because the way in which the data are prepared affects the results substantially.</p>
<p>Intuitively, there are two methods to vectorize the <code>caption</code>s:</p>
<ol>
<li>Each <code>caption</code> could be encoded in isolation with each one being treated as an entirely different field.</li>
<li>The <code>caption</code>s could all be concatenated and encoded together.</li>
</ol>
<p>Option #2 makes the most sense because there is no natural ordering of captions; for any given two users, there should be nothing in common between each of their first photo captions, or each of their second photo captions, etc.</p>
<p>It is possible that information can be lost by combining captions as in option #2. One example of this loss of data is when two photo captions with completely different sentiments are concatenated; however, the gender of the user who wrote the captions remains constant, which is the important detail.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [46]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="k">def</span> <span class="nf">extract_writing_example</span><span class="p">(</span><span class="n">row</span><span class="p">):</span>
    <span class="n">captions</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">medium</span> <span class="ow">in</span> <span class="n">row</span><span class="o">.</span><span class="n">media</span><span class="p">[</span><span class="s1">'nodes'</span><span class="p">]:</span>
        <span class="k">if</span> <span class="s1">'caption'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">medium</span><span class="p">:</span>
            <span class="k">continue</span>
            
        <span class="n">caption</span> <span class="o">=</span> <span class="n">medium</span><span class="p">[</span><span class="s1">'caption'</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">caption</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="n">captions</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">caption</span><span class="p">)</span>

    <span class="k">return</span> <span class="s1">' '</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">captions</span><span class="p">)</span>
        
<span class="n">data</span><span class="p">[</span><span class="s1">'writing_example'</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">extract_writing_example</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">data</span><span class="p">[[</span><span class="s1">'username'</span><span class="p">,</span> <span class="s1">'writing_example'</span><span class="p">]]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[46]:</div>
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
<th>username</th>
<th>writing_example</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>abigail13d</td>
<td></td>
</tr>
<tr>
<th>1</th>
<td>jkst0329</td>
<td></td>
</tr>
<tr>
<th>2</th>
<td>nicobonta18</td>
<td>Se tocchi la squadra giuro non torni a casa!😈😤&#8230;</td>
</tr>
<tr>
<th>3</th>
<td>sunnykumar7094</td>
<td></td>
</tr>
<tr>
<th>4</th>
<td>purbashamalakar</td>
<td>Some people ask me this is my fake profile but&#8230;</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Prepare-the-hash_tags-field">Prepare the <code>hash_tags</code> field<a class="anchor-link" href="#Prepare-the-hash_tags-field">¶</a></h3>
<p>This section may seem unnecessary because hash tags will already be detected and appropriately prioritized by the <code>writing_example</code> vectorizer. The impetus for synthesizing the <code>hash_tags</code> field is to be able to apply additional constraints on its vectorizer. One example of a beneficial tuning is binarizing the field instead of retaining the amount of times that the hash tag exists in a given <code>writing_example</code>.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [47]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">import</span> <span class="nn">re</span>

<span class="k">def</span> <span class="nf">extract_hash_tags</span><span class="p">(</span><span class="n">row</span><span class="p">):</span>
    <span class="n">hash_tags</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">findall</span><span class="p">(</span><span class="s1">'#[a-zA-Z]*'</span><span class="p">,</span> <span class="n">row</span><span class="p">[</span><span class="s1">'writing_example'</span><span class="p">])</span>
    
    <span class="n">hash_tags</span> <span class="o">=</span> <span class="p">[</span><span class="n">hash_tag</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">'#'</span><span class="p">,</span> <span class="s1">''</span><span class="p">)</span> <span class="k">for</span> <span class="n">hash_tag</span> <span class="ow">in</span> <span class="n">hash_tags</span><span class="p">]</span>
    
    <span class="k">return</span> <span class="s1">' '</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">hash_tags</span><span class="p">)</span>

<span class="n">data</span><span class="p">[</span><span class="s1">'hash_tags'</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">extract_hash_tags</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">data</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">'hash_tags'</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="nb">len</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">)][[</span><span class="s1">'writing_example'</span><span class="p">,</span> <span class="s1">'hash_tags'</span><span class="p">]]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[47]:</div>
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
<th>writing_example</th>
<th>hash_tags</th>
</tr>
</thead>
<tbody>
<tr>
<th>4</th>
<td>Some people ask me this is my fake profile but&#8230;</td>
<td>trust</td>
</tr>
<tr>
<th>5</th>
<td>Super excited to move to a new city, but not l&#8230;</td>
<td>plantlife plantlady plantlife diyjewelry dewyn&#8230;</td>
</tr>
<tr>
<th>6</th>
<td>I think my cat wants to go on a vacation 😅😅😅 #&#8230;</td>
<td>cats catsofinstagram cats catsoninstagram vaca&#8230;</td>
</tr>
<tr>
<th>7</th>
<td>Vegan Peanut Butter Banana Split! Organic vega&#8230;</td>
<td>hurricaneirma hurricaneirma hurricaneirma vega&#8230;</td>
</tr>
<tr>
<th>8</th>
<td>Nyder solnedgang på hørhus kollegiet #hollywoo&#8230;</td>
<td>hollywood walkoffame favoritehero grandcanyon &#8230;</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Prepare-the-first_name-field">Prepare the <code>first_name</code> field<a class="anchor-link" href="#Prepare-the-first_name-field">¶</a></h3>
<p>The <code>full_name</code> field is deceivingly unideal for use with bag-of-words encoding. Recall that 1-gram bag-of-words encoding discards information about where each word occurs in the text. Furthermore, people often have last names which could function as first names for the opposite gender, e.g. &#8220;Patricia James.&#8221; This scenario would be particularly bad if the weighting for the association of &#8220;James&#8221; with &#8220;male&#8221; were stronger than &#8220;Patricia&#8221;&#8216;s association with &#8220;female.&#8221;</p>
<p>To solve this problem, the <code>first_name</code> field is extracted and encoded separately as a crude approximation for the loss of positional information when encoding with bag-of-words. The use of n-grams does not solve this problem because very few people have the same <code>full_name</code>, so the model would be overfitted to the train data. The reason that the <code>first_name</code> field does not entirely replace the <code>full_name</code> field is that it may contain emojis or middle names with which predictions can be improved.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [48]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="k">def</span> <span class="nf">extract_first_name</span><span class="p">(</span><span class="n">row</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">row</span><span class="p">[</span><span class="s1">'full_name'</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">' '</span><span class="p">,</span> <span class="mi">1</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>

<span class="n">data</span><span class="p">[</span><span class="s1">'first_name'</span><span class="p">]</span> <span class="o">=</span> <span class="n">data</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">extract_first_name</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">data</span><span class="p">[[</span><span class="s1">'full_name'</span><span class="p">,</span> <span class="s1">'first_name'</span><span class="p">]]</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[48]:</div>
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
<th>full_name</th>
<th>first_name</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>Abigail Diaz Gamio 13🍁</td>
<td>Abigail</td>
</tr>
<tr>
<th>1</th>
<td>Jaronsa Taylor</td>
<td>Jaronsa</td>
</tr>
<tr>
<th>2</th>
<td>Nico Bontà😈</td>
<td>Nico</td>
</tr>
<tr>
<th>3</th>
<td>Sunny Kumar</td>
<td>Sunny</td>
</tr>
<tr>
<th>4</th>
<td>purbasha&#8230;..</td>
<td>purbasha&#8230;..</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Data-Exploration">Data Exploration<a class="anchor-link" href="#Data-Exploration">¶</a></h2>
<p>The section that follows will more deeply explore the dataset to identify the data&#8217;s features and trends. The goal of this investigation is to identify which encodings are optimal for analysis.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Sample-the-fully-formatted-data">Sample the fully formatted data<a class="anchor-link" href="#Sample-the-fully-formatted-data">¶</a></h3>
<p>Looking at a subset of the dataset with all of the fields properly formatted is the best way to spot obvious relationships and potential paths forward.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [49]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[[</span><span class="s1">'username'</span><span class="p">,</span> <span class="s1">'first_name'</span><span class="p">,</span> <span class="s1">'full_name'</span><span class="p">,</span> <span class="s1">'biography'</span><span class="p">,</span> <span class="s1">'writing_example'</span><span class="p">,</span> <span class="s1">'hash_tags'</span><span class="p">,</span> <span class="s1">'gender'</span><span class="p">,</span> <span class="s1">'gender_enc'</span><span class="p">]]</span>
<span class="n">data</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[49]:</div>
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
<th>username</th>
<th>first_name</th>
<th>full_name</th>
<th>biography</th>
<th>writing_example</th>
<th>hash_tags</th>
<th>gender</th>
<th>gender_enc</th>
</tr>
</thead>
<tbody>
<tr>
<th>0</th>
<td>abigail13d</td>
<td>Abigail</td>
<td>Abigail Diaz Gamio 13🍁</td>
<td>Just me. 19\nsnapchat: abbipandi</td>
<td></td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>1</th>
<td>jkst0329</td>
<td>Jaronsa</td>
<td>Jaronsa Taylor</td>
<td>Just a 23 year old living in Milwaukee</td>
<td></td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>2</th>
<td>nicobonta18</td>
<td>Nico</td>
<td>Nico Bontà😈</td>
<td>✖️⚠️Follow @billz433⚠️✖️</td>
<td>Se tocchi la squadra giuro non torni a casa!😈😤&#8230;</td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>3</th>
<td>sunnykumar7094</td>
<td>Sunny</td>
<td>Sunny Kumar</td>
<td></td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>4</th>
<td>purbashamalakar</td>
<td>purbasha&#8230;..</td>
<td>purbasha&#8230;..</td>
<td>Don&#8217;t worry be happy☺\n🎉wish me 23 Feb🎂\nBigge&#8230;</td>
<td>Some people ask me this is my fake profile but&#8230;</td>
<td>trust</td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>5</th>
<td>chelsea_a_bear</td>
<td>Chelsea</td>
<td>Chelsea Hebert</td>
<td></td>
<td>Super excited to move to a new city, but not l&#8230;</td>
<td>plantlife plantlady plantlife diyjewelry dewyn&#8230;</td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>6</th>
<td>jimsansa</td>
<td>Jim</td>
<td>Jim Van Mourik</td>
<td></td>
<td>I think my cat wants to go on a vacation 😅😅😅 #&#8230;</td>
<td>cats catsofinstagram cats catsoninstagram vaca&#8230;</td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>7</th>
<td>jnlfunfitfoodie</td>
<td>Jennifer</td>
<td>Jennifer Nicole Lee</td>
<td>&#8220;Happiest Woman Alive!&#8221; Blessed to motivate al&#8230;</td>
<td>Vegan Peanut Butter Banana Split! Organic vega&#8230;</td>
<td>hurricaneirma hurricaneirma hurricaneirma vega&#8230;</td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>8</th>
<td>peter_jordt</td>
<td>Peter</td>
<td>Peter Jordt</td>
<td></td>
<td>Nyder solnedgang på hørhus kollegiet #hollywoo&#8230;</td>
<td>hollywood walkoffame favoritehero grandcanyon &#8230;</td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>9</th>
<td>_d.kolobova_</td>
<td>Darina</td>
<td>Darina Kolobova</td>
<td></td>
<td>Огромное спасибо за эти 2 незабываемых недели😘&#8230;</td>
<td>malenkayastrana malenkayastrana scetchbook ma&#8230;</td>
<td>female</td>
<td>1</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Several patterns are immediately apparent in this dataset:</p>
<ol>
<li><em>Many users make liberal use of emojis</em>. One path forward is to perform character vectorization of all fields in which users may enter emojis.</li>
<li><em>Users write freeform text with context</em>. It may be beneficial to encode any user-inputted fields as n-grams rather than the default 1-grams to retain this context.</li>
<li><em>Not everyone has a <code>writing_example</code>, but almost everyone has filled in at least one text field.</em> This observation is good news for the model; users without any user-entered text fields filled in are typically less useful for marketing purposes.</li>
</ol>
<p>These observations could be validated by checking their correctness statistically instead of visually from the small sample of ten examples.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Chart-the-frequency-of-word-counts-for-each-field-and-each-gender">Chart the frequency of word counts for each field and each gender<a class="anchor-link" href="#Chart-the-frequency-of-word-counts-for-each-field-and-each-gender">¶</a></h3>
<p>Following observation #3 above, charting the frequency with which users of each gender fill in text-based fields will give some idea as to how reliable the use of these fields would be.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [50]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">field_lens_to_plot</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'writing_example'</span><span class="p">,</span> <span class="s1">'biography'</span><span class="p">,</span> <span class="s1">'full_name'</span><span class="p">,</span> <span class="s1">'hash_tags'</span><span class="p">]</span>

<span class="k">for</span> <span class="n">field</span> <span class="ow">in</span> <span class="n">field_lens_to_plot</span><span class="p">:</span>
    <span class="k">for</span> <span class="n">gender</span> <span class="ow">in</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">]:</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
        <span class="n">gender_data</span> <span class="o">=</span> <span class="n">data</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">]</span> <span class="o">==</span> <span class="n">gender</span><span class="p">]</span>
        <span class="n">gender_data</span><span class="p">[</span><span class="s2">"</span><span class="si">%s</span><span class="s2">_len"</span> <span class="o">%</span> <span class="n">field</span><span class="p">]</span> <span class="o">=</span> <span class="n">gender_data</span><span class="p">[</span><span class="n">field</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span>
            <span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="nb">len</span><span class="p">(</span><span class="n">re</span><span class="o">.</span><span class="n">findall</span><span class="p">(</span><span class="sa">r</span><span class="s1">'\w+'</span><span class="p">,</span> <span class="n">x</span><span class="p">)))</span>
        <span class="n">gender_data</span><span class="p">[</span><span class="s2">"</span><span class="si">%s</span><span class="s2">_len"</span> <span class="o">%</span> <span class="n">field</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span>
            <span class="n">title</span><span class="o">=</span><span class="s2">"Frequency of </span><span class="si">%s</span><span class="s2"> for </span><span class="si">%s</span><span class="s2">"</span> <span class="o">%</span> <span class="p">(</span><span class="n">field</span><span class="p">,</span> <span class="s1">'males'</span> <span class="k">if</span> <span class="n">gender</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="s1">'females'</span><span class="p">),</span>
            <span class="n">kind</span><span class="o">=</span><span class="s1">'hist'</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">'blue'</span> <span class="k">if</span> <span class="n">gender</span> <span class="o">==</span> <span class="mi">0</span> <span class="k">else</span> <span class="s1">'red'</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>/usr/local/lib/python3.6/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  
</pre>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The only field with a substantial discrepancy between genders is <code>biography</code>—females seem to be more likely to enter more text. Unfortunately, this finding was a red herring; when the trained model included the length of the <code>biography</code> field as a predictor, there was no significant difference in predictive power.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Check-for-imbalances-in-the-genders">Check for imbalances in the genders<a class="anchor-link" href="#Check-for-imbalances-in-the-genders">¶</a></h3>
<p>A substantial imbalance in the data may require intervention.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [51]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="s2">"Number of males: </span><span class="si">%d</span><span class="s2">; Number of females: </span><span class="si">%d</span><span class="s2">"</span> <span class="o">%</span> <span class="p">(</span>
    <span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]),</span>
    <span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">[</span><span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span><span class="p">])</span>
<span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[51]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>'Number of males: 8909; Number of females: 11709'</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>There is an imbalance in gender representations within the dataset, but the lopsidedness is insufficient to warrant drastic measures. One way in which the analysis could be made more robust is by using the <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic">AUROC metric</a> in place of accuracy for model optimization. This technique is typically used to compensate for acute asymmetry in the data, but it can also be employed for less extreme corrections. One challenge in the use of AUROC is that it is limited to binary classification, which limits the ability of the model to be extended later to support more than the binary genders.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Plot-the-most-predictive-words-for-each-field">Plot the most predictive words for each field<a class="anchor-link" href="#Plot-the-most-predictive-words-for-each-field">¶</a></h3>
<p>In this section, untuned logistic regression models will be trained on each field in isolation, and the most extreme weights outputted in graphical format. This illustration is not particularly useful or actionable, but it is interesting.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [52]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>

<span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">CountVectorizer</span>

<span class="kn">from</span> <span class="nn">model_performance_plotter</span> <span class="k">import</span> <span class="n">plot_coefficients</span>

<span class="n">plot_coefficients</span><span class="p">(</span><span class="n">LogisticRegression</span><span class="p">(),</span> <span class="n">CountVectorizer</span><span class="p">(),</span>
                  <span class="s1">'Biography Most Predictive Terms'</span><span class="p">,</span>
                  <span class="n">data</span><span class="p">[</span><span class="s1">'biography'</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>
<span class="n">plot_coefficients</span><span class="p">(</span><span class="n">LogisticRegression</span><span class="p">(),</span> <span class="n">CountVectorizer</span><span class="p">(),</span>
                  <span class="s1">'Writing Example Most Predictive Terms'</span><span class="p">,</span>
                  <span class="n">data</span><span class="p">[</span><span class="s1">'writing_example'</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>
<span class="n">plot_coefficients</span><span class="p">(</span><span class="n">LogisticRegression</span><span class="p">(),</span> <span class="n">CountVectorizer</span><span class="p">(),</span>
                  <span class="s1">'Full Name Most Predictive Terms'</span><span class="p">,</span>
                  <span class="n">data</span><span class="p">[</span><span class="s1">'full_name'</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>
<span class="n">plot_coefficients</span><span class="p">(</span><span class="n">LogisticRegression</span><span class="p">(),</span> <span class="n">CountVectorizer</span><span class="p">(),</span>
                  <span class="s1">'Hash Tags Most Predictive Terms'</span><span class="p">,</span>
                  <span class="n">data</span><span class="p">[</span><span class="s1">'hash_tags'</span><span class="p">],</span> <span class="n">data</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Model-Training-and-Validation">Model Training and Validation<a class="anchor-link" href="#Model-Training-and-Validation">¶</a></h2>
<p>The final step is to train and validate the model. In practice, this section took many iterations to get to where it now is.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Split-the-main-dataset-into-training-and-test-datasets">Split the main dataset into training and test datasets<a class="anchor-link" href="#Split-the-main-dataset-into-training-and-test-datasets">¶</a></h3>
<p>SciKit will automatically sample from the training dataset to create a cross-validation dataset, so only the test dataset must be created manually.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [53]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>
<span class="n">train</span><span class="p">,</span> <span class="n">test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Train data set size: </span><span class="si">%d</span><span class="s2">"</span> <span class="o">%</span> <span class="nb">len</span><span class="p">(</span><span class="n">train</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Test data set size: </span><span class="si">%d</span><span class="s2">"</span> <span class="o">%</span> <span class="nb">len</span><span class="p">(</span><span class="n">test</span><span class="p">))</span>
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
<pre>Train data set size: 16494
Test data set size: 4124
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [54]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">parameters</span> <span class="o">=</span> <span class="p">{</span>
    <span class="c1"># 'clf__solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga']</span>
    <span class="c1"># 'clf__loss': ['squared_hinge', 'hinge'],</span>
    <span class="c1"># 'clf__penalty': ['l1', 'l2'],</span>
    <span class="c1"># 'clf__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],</span>
    <span class="c1"># 'clf__C': [10, 15, 20, 25, 30, 35, 50, 80, 100, 120, 150],</span>
    <span class="c1"># 'clf__dual': [False, True],</span>
    <span class="c1"># 'clf__class_weight': [None, 'balanced'],</span>
<span class="p">}</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [56]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.pipeline</span> <span class="k">import</span> <span class="n">Pipeline</span>

<span class="kn">from</span> <span class="nn">sklearn.pipeline</span> <span class="k">import</span> <span class="n">FeatureUnion</span>

<span class="kn">from</span> <span class="nn">transformers</span> <span class="k">import</span> <span class="n">ItemSelector</span>
<span class="kn">from</span> <span class="nn">sklearn.feature_extraction.text</span> <span class="k">import</span> <span class="n">TfidfTransformer</span>
<span class="kn">from</span> <span class="nn">sklearn.feature_extraction</span> <span class="k">import</span> <span class="n">DictVectorizer</span>

<span class="n">tfidf_transformer</span> <span class="o">=</span> <span class="n">TfidfTransformer</span><span class="p">()</span>

<span class="n">encoding_args</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'decode_error'</span><span class="p">:</span> <span class="s1">'replace'</span><span class="p">,</span>
    <span class="s1">'strip_accents'</span><span class="p">:</span> <span class="s1">'unicode'</span><span class="p">,</span>
<span class="p">}</span>

<span class="n">word_vectorizer_args</span> <span class="o">=</span> <span class="p">{</span>
    <span class="o">**</span><span class="n">encoding_args</span><span class="p">,</span>
    <span class="s1">'ngram_range'</span><span class="p">:</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">)</span>
<span class="p">}</span>

<span class="n">char_vectorizer_args</span> <span class="o">=</span> <span class="p">{</span>
    <span class="o">**</span><span class="n">encoding_args</span><span class="p">,</span>
    <span class="s1">'analyzer'</span><span class="p">:</span> <span class="s1">'char'</span><span class="p">,</span>
    <span class="s1">'ngram_range'</span><span class="p">:</span> <span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span>
<span class="p">}</span>

<span class="n">word_vectorizer</span> <span class="o">=</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="o">**</span><span class="n">word_vectorizer_args</span><span class="p">)</span>
<span class="n">char_vectorizer</span> <span class="o">=</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="o">**</span><span class="n">char_vectorizer_args</span><span class="p">)</span>

<span class="n">transformers</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">'username'</span><span class="p">:</span> <span class="p">{</span>
        <span class="s1">'char'</span><span class="p">:</span> <span class="n">char_vectorizer</span>
    <span class="p">},</span>
    <span class="s1">'biography'</span><span class="p">:</span> <span class="p">{</span>
        <span class="s1">'word'</span><span class="p">:</span> <span class="n">word_vectorizer</span><span class="p">,</span>
        <span class="s1">'char'</span><span class="p">:</span> <span class="n">char_vectorizer</span>
    <span class="p">},</span>
    <span class="s1">'full_name'</span><span class="p">:</span> <span class="p">{</span>
        <span class="s1">'word'</span><span class="p">:</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="o">**</span><span class="n">encoding_args</span><span class="p">),</span>
        <span class="s1">'char'</span><span class="p">:</span> <span class="n">char_vectorizer</span>
    <span class="p">},</span>
    <span class="s1">'first_name'</span><span class="p">:</span> <span class="p">{</span>
        <span class="s1">'word'</span><span class="p">:</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="o">**</span><span class="n">encoding_args</span><span class="p">)</span>
    <span class="p">},</span>
    <span class="s1">'hash_tags'</span><span class="p">:</span> <span class="p">{</span>
        <span class="s1">'word'</span><span class="p">:</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="o">**</span><span class="n">encoding_args</span><span class="p">,</span> <span class="n">binary</span><span class="o">=</span><span class="kc">True</span><span class="p">),</span>
        <span class="s1">'char'</span><span class="p">:</span> <span class="n">CountVectorizer</span><span class="p">(</span><span class="o">**</span><span class="n">char_vectorizer_args</span><span class="p">,</span> <span class="n">binary</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="p">},</span>
    <span class="s1">'writing_example'</span><span class="p">:</span> <span class="p">{</span>
        <span class="s1">'word'</span><span class="p">:</span> <span class="n">word_vectorizer</span><span class="p">,</span>
        <span class="s1">'char'</span><span class="p">:</span> <span class="n">char_vectorizer</span>
    <span class="p">}</span>
<span class="p">}</span>

<span class="n">transformer_list</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">key</span><span class="p">,</span> <span class="n">transformer_types</span> <span class="ow">in</span> <span class="n">transformers</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
    <span class="k">for</span> <span class="n">transformer_type</span><span class="p">,</span> <span class="n">transformer</span> <span class="ow">in</span> <span class="n">transformer_types</span><span class="o">.</span><span class="n">items</span><span class="p">():</span>
        <span class="n">transformer_list</span><span class="o">.</span><span class="n">append</span><span class="p">(</span>
            <span class="p">(</span><span class="s2">"</span><span class="si">%s</span><span class="s2">_</span><span class="si">%s</span><span class="s2">"</span> <span class="o">%</span> <span class="p">(</span><span class="n">key</span><span class="p">,</span> <span class="n">transformer_type</span><span class="p">),</span> <span class="n">Pipeline</span><span class="p">([</span>
                <span class="p">(</span><span class="s1">'selector'</span><span class="p">,</span> <span class="n">ItemSelector</span><span class="p">(</span><span class="n">key</span><span class="o">=</span><span class="n">key</span><span class="p">)),</span>
                <span class="p">(</span><span class="s1">'vect'</span><span class="p">,</span> <span class="n">transformer</span><span class="p">),</span>
                <span class="p">(</span><span class="s1">'tfidf'</span><span class="p">,</span> <span class="n">tfidf_transformer</span><span class="p">)</span>
            <span class="p">]))</span>
        <span class="p">)</span>

<span class="n">pipeline</span> <span class="o">=</span> <span class="n">Pipeline</span><span class="p">([</span>
    <span class="p">(</span><span class="s1">'union'</span><span class="p">,</span> <span class="n">FeatureUnion</span><span class="p">(</span><span class="n">transformer_list</span><span class="o">=</span><span class="n">transformer_list</span><span class="p">)),</span>
    <span class="p">(</span><span class="s1">'clf'</span><span class="p">,</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">150</span><span class="p">))</span>
<span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [57]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">accuracy_score</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">make_scorer</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">GridSearchCV</span>

<span class="n">scoring</span> <span class="o">=</span> <span class="p">{</span><span class="s1">'AUC'</span><span class="p">:</span> <span class="s1">'roc_auc'</span><span class="p">,</span> <span class="s1">'Accuracy'</span><span class="p">:</span> <span class="n">make_scorer</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">)}</span>
<span class="n">grid_search</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">pipeline</span><span class="p">,</span> <span class="n">parameters</span><span class="p">,</span> <span class="n">n_jobs</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">verbose</span><span class="o">=</span><span class="mi">10</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="n">scoring</span><span class="p">,</span> <span class="n">refit</span><span class="o">=</span><span class="s1">'AUC'</span><span class="p">)</span>
<span class="n">grid_search</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">train</span><span class="p">,</span> <span class="n">train</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"Best score: </span><span class="si">%0.3f</span><span class="s2">"</span> <span class="o">%</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">best_score_</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Best parameters set:"</span><span class="p">)</span>

<span class="n">best_parameters</span> <span class="o">=</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">best_estimator_</span><span class="o">.</span><span class="n">get_params</span><span class="p">()</span>
<span class="k">for</span> <span class="n">param_name</span> <span class="ow">in</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">parameters</span><span class="o">.</span><span class="n">keys</span><span class="p">()):</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\t</span><span class="si">%s</span><span class="s2">: </span><span class="si">%r</span><span class="s2">"</span> <span class="o">%</span> <span class="p">(</span><span class="n">param_name</span><span class="p">,</span> <span class="n">best_parameters</span><span class="p">[</span><span class="n">param_name</span><span class="p">]))</span>

<span class="n">score</span> <span class="o">=</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">test</span><span class="p">,</span> <span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Test score: </span><span class="si">%f</span><span class="s2">"</span> <span class="o">%</span> <span class="n">score</span><span class="p">)</span>

<span class="n">y_pred</span> <span class="o">=</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Test accuracy: </span><span class="si">%f</span><span class="s2">"</span> <span class="o">%</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">],</span> <span class="n">y_pred</span><span class="p">))</span>

<span class="c1"># Use this to assess the probability of each classification.</span>
<span class="c1"># grid_search.predict_proba(test)</span>

<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">classification_report</span>
<span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">],</span> <span class="n">y_pred</span><span class="p">))</span>
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
<pre>Fitting 3 folds for each of 1 candidates, totalling 3 fits
[CV]  ................................................................
[CV]  ................................................................
[CV]  ................................................................
[CV]  , AUC=0.9606773914601812, Accuracy=0.8974358974358975, total= 1.1min
[CV]  , AUC=0.9595042119976971, Accuracy=0.891778828664969, total= 1.1min
[CV]  , AUC=0.9536360530299655, Accuracy=0.8863016190649445, total= 1.1min
</pre>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stderr output_text">
<pre>[Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:  2.0min remaining:    0.0s
[Parallel(n_jobs=-1)]: Done   3 out of   3 | elapsed:  2.0min finished
</pre>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Best score: 0.958
Best parameters set:
Test score: 0.959340
Test accuracy: 0.892338
             precision    recall  f1-score   support

          0       0.90      0.85      0.87      1821
          1       0.88      0.93      0.91      2303

avg / total       0.89      0.89      0.89      4124

</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [58]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">model_performance_plotter</span> <span class="k">import</span> <span class="n">plot_learning_curve</span><span class="p">,</span> \
                                      <span class="n">plot_roc_curve</span><span class="p">,</span> \
                                      <span class="n">plot_precision_recall_curve</span>

<span class="n">title</span> <span class="o">=</span> <span class="s1">'Gender Classifier'</span>

<span class="n">plot_roc_curve</span><span class="p">(</span><span class="n">title</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>

<span class="n">plot_precision_recall_curve</span><span class="p">(</span><span class="n">title</span><span class="p">,</span> <span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">],</span> <span class="n">grid_search</span><span class="o">.</span><span class="n">decision_function</span><span class="p">(</span><span class="n">test</span><span class="p">))</span>

<span class="n">plot_learning_curve</span><span class="p">(</span><span class="n">grid_search</span><span class="o">.</span><span class="n">best_estimator_</span><span class="p">,</span> <span class="n">title</span><span class="p">,</span> <span class="n">train</span><span class="p">,</span> <span class="n">train</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">])</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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ZD9JrFmzBoDVq1ez0047ASHR3HzzzQAceOCBHJWmD7KpTBSLgJ0THneOtlW3T6mkFkA7YHkKY3LONTFPP/00Q4YMYbfddqOgoIBZs2bRv3//pK+ZN28eXbp0oW3btrUe/8ILL2Tq1KmbbR82bBiXXbbpB85Fixax7777fvu4c+fOLFpUtViECRMmcNhhh7HlllvStm1bpk2bBkDfvn154oknOP/883nyySf5/PPPWb58OQUFmzXENKpUJoqZQA9J3QgJYRhwQpV9JgOnAq8DxwIvmlnKawzOufSr7ZN/qkyaNInzzz8fCIX3pEmT6N+/f42jg+raTXrLLbc0OMbqjjllyhQGDhzI2LFjueiii5gwYQI33ngjo0aNYuLEiey///506tSJ5tVOH9G4UpYozKxc0ijgOaA58DczK5Z0LVBkZpOBvwL3SZoHrCAkE+ecaxQrVqzgxRdfZPbs2UiioqICSYwdO5aCggJWrly52f7t27ene/fufPLJJ6xZs6bWWkVdahSdOnVi4cKNXbelpaV06rRp121ZWRlvv/02AwcOBOC4445jyJAhAOy000488cQTQGhSe/zxx9l2221jvhsNYGY59dW/f39zzuWGkpKSjJ7/L3/5i40cOXKTbfvvv7+9/PLLtm7dOuvateu3MS5YsMC6dOliq1atMjOzSy65xIYPH25ff/21mZktXbrUHnnkkQbFM2fOHOvTp4+tW7fO5s+fb926dbPy8vJN9lm/fr0VFBTY3LlzzcxswoQJdvTRR5uZWVlZmVVUVJiZ2W9/+1u78sorqz1Pde874QN6vcpdv7XYOZe3Jk2axM9//vNNth1zzDFMmjSJLbbYgvvvv5/TTjuNvffem2OPPZYJEybQrl07AMaMGUOHDh3o3bs3e+65J4cffnisPotk9thjD4YOHUrv3r0ZMmQI48aN+7bp6LDDDuPTTz+lRYsW3HPPPRxzzDH07duX++67j7FjxwJhSG/Pnj3ZbbfdWLJkCaNHj25QPHHJcqxLoLCw0IqKijIdhnMuhnfffZdevXplOowmp7r3XdIsMyusz/G8RuGccy4pTxTOOeeS8kThnEupXGveznWpeL89UTjnUqZ169YsX77ck0WaWLQeRetGXo45J6bwcM7lps6dO1NaWko65mhzQeUKd43JE4VzLmVatmzZqCutuczwpifnnHNJeaJw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" /></div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
<div class="output_area">
<div class="prompt output_prompt">Out[58]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>&lt;module 'matplotlib.pyplot' from '/usr/local/lib/python3.6/site-packages/matplotlib/pyplot.py'&gt;</pre>
</div>
</div>
<div class="output_area">
<div class="prompt"></div>
<div class="output_png output_subarea "><img decoding="async" src="data:image/png;base64,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" /></div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Examine-cases-where-the-model-makes-correct-predictions">Examine cases where the model makes correct predictions<a class="anchor-link" href="#Examine-cases-where-the-model-makes-correct-predictions">¶</a></h3>
<p>It is good practice to verify that the model is making reasonable predictions and that the labels were accurate.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [59]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">test</span><span class="p">[</span><span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">]</span> <span class="o">==</span> <span class="n">y_pred</span><span class="p">]</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[59]:</div>
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
<th>username</th>
<th>first_name</th>
<th>full_name</th>
<th>biography</th>
<th>writing_example</th>
<th>hash_tags</th>
<th>gender</th>
<th>gender_enc</th>
</tr>
</thead>
<tbody>
<tr>
<th>1365</th>
<td>_kim_law</td>
<td>stacey</td>
<td>stacey</td>
<td>🇯🇲Proud Jamaican&#8230;Island girl🌴🇯🇲</td>
<td></td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>4085</th>
<td>katiebiancaxox</td>
<td>Katie</td>
<td>Katie Bianca</td>
<td>3 soon to be 4 ❤️</td>
<td></td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>3968</th>
<td>jon.bon.jovi_always</td>
<td>immortal</td>
<td>immortal rock 🎸☇🔥🤘</td>
<td>&#8220;Shot through the heart \nAnd you&#8217;re to blame &#8230;</td>
<td>Jon&#8217;s original vocals only, isolated from the &#8230;</td>
<td>jonbonjovi bonjovi rock rockbands rockmusic ha&#8230;</td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>9788</th>
<td>ylimenarod</td>
<td>Emily</td>
<td>Emily Doran</td>
<td>悲しい女の子 ( ͡° ͜ʖ ͡°)\n@intotheshade_</td>
<td>Looks like one eyed Kenny\n#35mm #minoltax700 &#8230;</td>
<td>minoltax agfa minoltax agfa</td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>6936</th>
<td>sacredlotus17</td>
<td>Melissa</td>
<td>Melissa Pattinson</td>
<td></td>
<td>Here&#8217;s a sneak peek of what I&#8217;m working on at &#8230;</td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>5076</th>
<td>angelfesh744</td>
<td>Felicia</td>
<td>Felicia</td>
<td></td>
<td></td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>5356</th>
<td>prescott127</td>
<td>Prescott</td>
<td>Prescott</td>
<td>活在當下！！！ live in present!!!</td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>8681</th>
<td>tjmacca</td>
<td>Thomas</td>
<td>Thomas McKenzie</td>
<td></td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>2099</th>
<td>kellyhosy</td>
<td>Kelly</td>
<td>Kelly Ho</td>
<td>Join The Jobless Club and be fabulous</td>
<td>Sexercise the wall \n@verxniques 🤰🏻#rockclimbi&#8230;</td>
<td>rockclimbing exercise sexercise booty rockclim&#8230;</td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>1664</th>
<td>alfonso3892</td>
<td>Alfonso</td>
<td>Alfonso Martinez</td>
<td></td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
</tbody>
</table>
</div>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Examine-cases-where-the-model-makes-incorrect-predictions">Examine cases where the model makes incorrect predictions<a class="anchor-link" href="#Examine-cases-where-the-model-makes-incorrect-predictions">¶</a></h3>
<p>It is also good practice to investigate the cases for which the model makes incorrect predictions. Note that in the list below, the <code>gender</code> field is the true label, and the opposite of this label is what the model predicted. The majority of these mistakes are due to incorrect labels.</p>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In [62]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="n">test</span><span class="p">[</span><span class="n">test</span><span class="p">[</span><span class="s1">'gender_enc'</span><span class="p">]</span> <span class="o">!=</span> <span class="n">y_pred</span><span class="p">]</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="mi">10</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[62]:</div>
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
<th>username</th>
<th>first_name</th>
<th>full_name</th>
<th>biography</th>
<th>writing_example</th>
<th>hash_tags</th>
<th>gender</th>
<th>gender_enc</th>
</tr>
</thead>
<tbody>
<tr>
<th>9734</th>
<td>yigitsun97</td>
<td></td>
<td></td>
<td>Sanity is in the eye of the beholder.</td>
<td>Philoxene Iskeleden bir cisim yaklasiyor kapta&#8230;</td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>10102</th>
<td>ztingle17</td>
<td>Zachry</td>
<td>Zachry Ray Tingle</td>
<td>Texas A&amp;M University Class of 2018 👍🏻 Basketba&#8230;</td>
<td>Great weekend with the family! Glad my cousins&#8230;</td>
<td>NationalBestFriendDay NationalSiblingDay MyGor&#8230;</td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>7412</th>
<td>gorjuszj</td>
<td>Shae</td>
<td>Shae</td>
<td>SNAP ME 👻 [GORJUSZ]. ♈️3/25. Raising a Princes&#8230;</td>
<td>♥️ New hair who dis ? 📞!\n#autumnhair #fallhai&#8230;</td>
<td>autumnhair fallhair naturalhair bob dallasstyl&#8230;</td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>4982</th>
<td>khor_meng_yang107</td>
<td>Khor</td>
<td>Khor Meng Yang</td>
<td>🏣SMK MIHARJA \n💒†FGA CYC\n🎂1007\nWC :khor1234&#8230;</td>
<td>Today 🌝#04132017 Today SPAM吗😎</td>
<td></td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>1992</th>
<td>pouria_.rs</td>
<td></td>
<td></td>
<td>#BhMn\n#Я§ons\n❤👑</td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>3058</th>
<td>dinarinus</td>
<td>Din</td>
<td>Din Arinus</td>
<td></td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>621</th>
<td>viv.ek.5203</td>
<td>Vîv</td>
<td>Vîv Ek</td>
<td>Simply luvable</td>
<td></td>
<td></td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>3963</th>
<td>santannasheneal</td>
<td>Signature</td>
<td>Signature by Santanna Sheneal</td>
<td>Makeup Artist/Esthetician 🎨\nXtreme Lash Speci&#8230;</td>
<td>The moment I&#8217;ve been waiting for. 😍😍😍\nThe onl&#8230;</td>
<td>Beauty Bar Supply MakeupArtist Braiders Beauti&#8230;</td>
<td>male</td>
<td>0</td>
</tr>
<tr>
<th>1607</th>
<td>tmongram</td>
<td>Gabriele</td>
<td>Gabriele Beddoni △</td>
<td>Rome,Paris⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Illustr&#8230;</td>
<td>⠀⠀⠀ 1 3 windows ⠀⠀⠀⠀ ⠀⠀⠀⠀⠀ #picoftheday #inst&#8230;</td>
<td>picoftheday instagood instamood goodvibes sony&#8230;</td>
<td>female</td>
<td>1</td>
</tr>
<tr>
<th>441</th>
<td>symiko70</td>
<td>💖Symiko💖</td>
<td>💖Symiko💖</td>
<td>👬Loving My 2 Boys👬</td>
<td></td>
<td></td>
<td>female</td>
<td>1</td>
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
<div class="prompt input_prompt">In [20]:</div>
<div class="inner_cell">
<div class="input_area">
<div class=" highlight hl-ipython3">
<pre><span class="kn">from</span> <span class="nn">sklearn.externals</span> <span class="k">import</span> <span class="n">joblib</span>

<span class="n">MODEL_FILE</span> <span class="o">=</span> <span class="s1">'ig_gender_classifier.pkl'</span>
<span class="n">joblib</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="n">grid_search</span><span class="p">,</span> <span class="n">MODEL_FILE</span><span class="p">)</span>
</pre>
</div>
</div>
</div>
</div>
<div class="output_wrapper">
<div class="output">
<div class="output_area">
<div class="prompt output_prompt">Out[20]:</div>
<div class="output_text output_subarea output_execute_result">
<pre>['ig_gender_classifier.pkl']</pre>
</div>
</div>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt"></div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion">¶</a></h2>
<p>The model achieves 90% accuracy with 90% precision and 85% recall for males, and 88% precision and 93% recall for females; therefore, it is slightly superior at picking out females.</p>
<p>In the future, this project could be improved in the following ways:</p>
<ul>
<li><em>Investigating why the model performs better on females than males</em>. One possible cause for this discrepancy is that there are more females in the dataset, so the model has more data with which to identify females.</li>
<li><em>Translating non-English text to English and then passing that through the model</em>. One way to look at translation is that it is a poor man&#8217;s form of PCA; the model could share the weights of English terms rather than being spread thin on every input language. This experiment was attempted but it was found to be too slow due the need for a web request for every example.</li>
<li><em>Redoing the project with a neural net instead of logistic regression</em>. Neural nets typically require at least 50,000 to 100,000 examples to perform substantially better than classical models. This experiment was attempted early on in the project, but failed due to an insufficient number of examples.</li>
<li><em>Incorporating user photos into the model via ensemble methods</em>. Computer vision is expensive and slow, so this addition is unlikely to add substantial value to the end result.</li>
</ul>
</div>
</div>
</div>

