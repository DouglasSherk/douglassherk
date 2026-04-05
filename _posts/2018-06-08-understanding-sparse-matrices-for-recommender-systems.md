---
layout: post
title: "Understanding compressed matrices for sparse data"
date: 2018-06-08
categories:
  - Data Engineering
  - Recommender Systems
source_url: https://datastronomy.com/understanding-sparse-matrices-for-recommender-systems/
---
<p>In this article, you&#8217;ll learn about how matrices representing sparse data, e.g. user ratings, can be compressed to vertically scale machine learning algorithms by fitting datasets into memory that would otherwise be too large. You&#8217;ll learn about the various compression formats and delve into their trade-offs.<br />
</p>
<div style="display: none;">delve into examples with SciPy</div>
<h1>Sparsity</h1>
<p>Collaborative filtering recommender systems are usually built using data from user interactions, and user interactions are inherently <strong>sparse</strong>. For example, only 0.941% of the ratings in the <a href="https://grouplens.org/datasets/movielens/">MovieLens dataset</a> are present [1].</p>
<p>In addition to the algorithmic challenges inherent to discovering the latent features of such incomplete data, another challenge is in scaling systems that accept memory-hungry matrix inputs.</p>
<p>Let&#8217;s anchor our discussion to an example. Assume that we&#8217;re working with a movie recommender model and that its input data is tabulated below.</p>
<p><figure id="attachment_265" aria-describedby="caption-attachment-265" style="width: 750px" class="wp-caption aligncenter"><img data-recalc-dims="1" loading="lazy" decoding="async" class="wp-image-265 size-large" src="https://i0.wp.com/www.datastronomy.com/wp-content/uploads/2018/04/Screen-Shot-2018-05-28-at-4.00.44-PM-1024x224.png?resize=750%2C164" alt="" width="750" height="164" /><figcaption id="caption-attachment-265" class="wp-caption-text">Cells with grey backgrounds and no numbers have no user ratings</figcaption></figure></p>
<h1>Dense Matrices Review</h1>
<p>When the above ratings are munged to be valid inputs for a dimensionality reduction algorithm such as SVD, every cell must be filled. Ignoring for now demeaning and standardization, one possible form of the input is as follows:</p>
<p class="ql-center-displayed-equation" style="line-height: 128px;"><span class="ql-right-eqno"> &nbsp; </span><span class="ql-left-eqno"> &nbsp; </span><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-01ac241f8be48a5d432aec518238c017_l3.png" height="128" width="122" class="ql-img-displayed-equation quicklatex-auto-format" alt="&#92;&#091; &#92;&#98;&#101;&#103;&#105;&#110;&#123;&#98;&#109;&#97;&#116;&#114;&#105;&#120;&#125; &#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#52;&#32;&#38;&#32;&#53;&#32;&#38;&#32;&#48;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#51;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#92;&#92; &#50;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#52;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#51; &#92;&#101;&#110;&#100;&#123;&#98;&#109;&#97;&#116;&#114;&#105;&#120;&#125; &#92;&#093;" title="Rendered by QuickLaTeX.com"/></p>
<p>Notice how zeros are filled in cells for which there is a lack of data. On such a small dataset, this representation only wastes a few dozen bytes at most, but it&#8217;s not difficult to see how the amount of wasted memory grows exponentially as a function of the numbers of users and features (<img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-306ba3d974b5cbd110589f3aa7b0a710_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#79;&#40;&#124;&#117;&#124;&#32;&#92;&#99;&#100;&#111;&#116;&#32;&#124;&#102;&#124;&#41;" title="Rendered by QuickLaTeX.com" height="19" width="80" style="vertical-align: -5px;"/>).</p>
<p class="ql-center-displayed-equation" style="line-height: 160px;"><span class="ql-right-eqno"> &nbsp; </span><span class="ql-left-eqno"> &nbsp; </span><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-92e84188fc81df1f4955451f64e359da_l3.png" height="160" width="159" class="ql-img-displayed-equation quicklatex-auto-format" alt="&#92;&#091; &#92;&#98;&#101;&#103;&#105;&#110;&#123;&#98;&#109;&#97;&#116;&#114;&#105;&#120;&#125; &#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#52;&#32;&#38;&#32;&#53;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#92;&#104;&#100;&#111;&#116;&#115;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#51;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#92;&#104;&#100;&#111;&#116;&#115;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#92;&#104;&#100;&#111;&#116;&#115;&#32;&#92;&#92; &#50;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#92;&#104;&#100;&#111;&#116;&#115;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#52;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#92;&#104;&#100;&#111;&#116;&#115;&#32;&#92;&#92; &#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#48;&#32;&#38;&#32;&#51;&#32;&#38;&#32;&#92;&#104;&#100;&#111;&#116;&#115;&#32;&#92;&#92; &#92;&#118;&#100;&#111;&#116;&#115;&#32;&#38;&#32;&#92;&#118;&#100;&#111;&#116;&#115;&#32;&#38;&#32;&#92;&#118;&#100;&#111;&#116;&#115;&#32;&#38;&#32;&#92;&#118;&#100;&#111;&#116;&#115;&#32;&#38;&#32;&#92;&#118;&#100;&#111;&#116;&#115;&#32;&#38;&#32;&#92;&#100;&#100;&#111;&#116;&#115; &#92;&#101;&#110;&#100;&#123;&#98;&#109;&#97;&#116;&#114;&#105;&#120;&#125; &#92;&#093;" title="Rendered by QuickLaTeX.com"/></p>
<p>Suppose that only 1% of this matrix is filled with ratings; then this representation consumes 100x more memory than it must.</p>
<p>The main advantage of uncompressed matrices is that access complexity is <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-d1a0977d9e713b6eb464199427db69ec_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#79;&#40;&#49;&#41;" title="Rendered by QuickLaTeX.com" height="19" width="36" style="vertical-align: -5px;"/> since cells are indexed statically using their row and column numbers. Uncompressed matrices are also easy to work with.</p>
<h1>Compression</h1>
<p>Many computer science problems involve the space-time complexity trade-off, a rule which describes how memory and CPU usage can be traded to achieve desirable performance characteristics. Memory is usually consumed to improve CPU usage, but the converse is equally valid.</p>
<p>The way to improve memory usage for the cost of more CPU is by <strong>compressing</strong> the matrices.</p>
<p><img data-recalc-dims="1" loading="lazy" decoding="async" class="aligncenter  wp-image-281" src="https://i0.wp.com/www.datastronomy.com/wp-content/uploads/2018/06/Compression-and-Sparsity.png?resize=345%2C345" alt="" width="345" height="345" /></p>
<p>The distinct terms &#8220;compressed&#8221; and &#8220;sparse&#8221; are often used interchangeably. &#8220;Sparse&#8221; refers to the nature of inputs and indicates that only an arbitrarily-sized minority of the data is known. &#8220;Compressed&#8221; matrices are stored in a format that requires preprocessing to be usable, and that ideally uses less memory than an uncompressed format.</p>
<p>A matrix can be uncompressed but sparse, as well as it can be compressed but dense, though these representations are both suboptimal.</p>
<h2>Compression vs. Dimensionality Reduction</h2>
<p>The astute reader may have noticed that dimensionality reduction algorithms such as PCA and SVD also compress their inputs. This observation is correct, but it misses an important distinction: dimensionality reduction is <strong>lossy</strong>, while compression as this article defines it is <strong>lossless</strong>.</p>
<p><figure id="attachment_268" aria-describedby="caption-attachment-268" style="width: 351px" class="wp-caption aligncenter"><img data-recalc-dims="1" loading="lazy" decoding="async" class="wp-image-268" src="https://i0.wp.com/www.datastronomy.com/wp-content/uploads/2018/04/Lossy-vs-Lossless-Compression.png?resize=351%2C258" alt="" width="351" height="258" /><figcaption id="caption-attachment-268" class="wp-caption-text">Lossy vs. lossless compression</figcaption></figure></p>
<p>Lossy compression irrecoverably loses some of the original data, while lossless compression can be perfectly undone. It&#8217;s analogous to the difference between encoding a high-quality recording of a song as an MP3 (lossy) vs. packing it into a zip file (lossless).</p>
<p>Another key difference is that lossily compressed data can often be used directly without any preprocessing, while losslessly compressed data must be unpacked before it can be used. The former should be apparent if you are familiar with dimensionality reduction techniques, while the latter will become clear as you read the remainder of this article.</p>
<h2>Formats</h2>
<p>There are many ways to compress matrices, but you only need to know a handful of them. The main formats are COOrdinate (COO) format, compressed sparse rows (CSR), and compressed sparse columns (CSC).</p>
<h3>COOrdinate (&#8220;COO&#8221;)</h3>
<p>SciPy documentation: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix">scipy.sparse.coo_matrix</a></p>
<p>Consider the most obvious compression scheme one: storing values in a series of zero-indexed <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-f2cc7fdbb9ba90fd8eac175adbd6ac10_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#40;&#120;&#44;&#32;&#121;&#41;" title="Rendered by QuickLaTeX.com" height="19" width="39" style="vertical-align: -5px;"/> tuples. That&#8217;s exactly what COOrdinate (&#8220;COO&#8221;) format is. The example that we&#8217;ve been working with is reformatted in this tuple format below.</p>
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-19115e4f95e8f24c2628afee9a7de6f8_l3.png" height="172" width="405" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>Although COO is not the most useful compressed matrix representation, it&#8217;s the most intuitive, and it&#8217;s quick and easy to convert back and forth from CSR and CSC formats.</p>
<h3>Compressed sparse rows (CSR)</h3>
<p>SciPy documentation: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html">scipy.sparse.csr_matrix</a></p>
<p>The compressed sparse row (CSR) representation is the most commonly used matrix compression scheme. While CSR is not as intuitive as COO, it makes up for this deficiency by being superior in almost every other way. CSR is fundamentally a further optimization to the COO format.</p>
<p>CSR breaks down its input matrix into three column vectors: <em>rows</em>, <em>columns</em>, and <em>values</em>. Other than the <em>rows</em> column, these vectors are identical to the columns of the COO format.</p>
<p>The <em>values</em> vector is a flattened list of the values present in the input matrix. In the example above, this vector corresponds with the <em>values</em> column of the COO table above:</p>
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-43838b0b54f6b444f474cae0d202d306_l3.png" height="161" width="384" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>Each of the elements of the <em>columns</em> vector indicate the column number into which the corresponding value in the <em>values</em> vector falls. Continuing with the main example in the article, this vector would be:</p>
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-d5f09a6a0b6b7286293680664e9fe6c6_l3.png" height="161" width="351" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>More concretely, the first value 2 indicates that the first value 4 of the <em>values</em> vector goes into the 3rd column (2 when zero-indexed).</p>
<p>The final vector is the one indicating the <em>rows</em>. Recall the <em>rows</em> column from the COO format example above:</p>
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-87e345f3b74566bf549803a7b8f748c3_l3.png" height="161" width="119" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>Since the cells from the original uncompressed matrix are read from left to right, then top to bottom, the numbers in the <em>x (rows)</em> column increase monotonically. Additionally, there&#8217;s redundancy as both the first (4) and second (5) values are on the first (0th) row.</p>
<p>Redundancy is a good litmus test for optimization. Rather than directly storing the row numbers of each value, CSR stores a separate contiguous array of the <em>{values, columns}</em> indexes at which new rows begin. An example is below.</p>
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-5d537854ce0c1bd0cd4139de0b209329_l3.png" height="161" width="477" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>Notice that the value 2 appears twice in the <em>r</em> vector. The row with <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-16ec1d81dc1a7d422c1985f813b6603b_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#105;&#32;&#61;&#32;&#50;" title="Rendered by QuickLaTeX.com" height="12" width="38" style="vertical-align: 0px;"/> contains no entries, so it is skipped in the compressed representation by immediately moving on to <img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-aff5f2775b2b4cf57abc99cdbb528d3c_l3.png" class="ql-img-inline-formula quicklatex-auto-format" alt="&#114;&#32;&#61;&#32;&#51;" title="Rendered by QuickLaTeX.com" height="12" width="41" style="vertical-align: 0px;"/> in the next entry.</p>
<p>In this example, the compressed row representation doesn&#8217;t save any memory, but if there were more values than rows, then it would.</p>
<div style="display: none;">
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-7239db5a20daad6e15c2a6afc5bdb733_l3.png" height="161" width="446" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>[latex]<br />
[+preamble]<br />
\usepackage{tikz}<br />
\usepackage{pgfplots}<br />
\pgfplotsset{compat=newest}<br />
[/preamble]<br />
\newcommand\tikznode[3][]%<br />
{\tikz[remember picture,baseline=(#2.base)]<br />
\node[minimum size=0pt,inner sep=0pt,#1](#2){#3};%<br />
}<br />
\newcommand{\tikzmark}[1]{\tikz[overlay, remember picture] \coordinate (#1);}<br />
\begin{bmatrix}<br />
\tikzmark{varrowtop} \tikzmark{harrowleft} 0 &amp; 0 &amp; 4 &amp; 5 &amp; 0 \tikzmark{harrowright} \\<br />
0 &amp; 3 &amp; 0 &amp; 0 &amp; 0 \\<br />
0 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\<br />
2 &amp; 0 &amp; 0 &amp; 0 &amp; 0 \\<br />
0 &amp; 4 &amp; 0 &amp; 0 &amp; 0 \\<br />
\tikzmark{varrowbottom} 0 &amp; 0 &amp; 0 &amp; 0 &amp; \tikznode{ref}{3}<br />
\end{bmatrix} \Rightarrow<br />
\begin{bmatrix}<br />
v \\ c<br />
\end{bmatrix} = \begin{bmatrix}<br />
4 &amp; 5 &amp; 3 &amp; 2 &amp; 4 &amp; \tikznode{val}{3} \\<br />
2 &amp; 3 &amp; 1 &amp; 0 &amp; 1 &amp; \tikznode{col}{4}<br />
\end{bmatrix}</p>
<p class="ql-center-picture"><img data-recalc-dims="1" loading="lazy" decoding="async" src="/assets/posts/understanding-sparse-matrices-for-recommender-systems/quicklatex.com-c1bfcb941082adbd60cd49ac12328d9b_l3.png" height="106" width="653" class="ql-img-picture quicklatex-auto-format" alt="Rendered by QuickLaTeX.com" title="Rendered by QuickLaTeX.com"/></p>
<p>[/latex]</p></div>
<h3>Compressed sparse columns (CSC)</h3>
<p>SciPy documentation: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html">scipy.sparse.csc_matrix</a></p>
<p>The compressed sparse column (CSC) scheme is virtually identical to CSR. The only difference is that the <em>columns</em> vector is compressed, while the <em>rows</em> vector is identical to the one in the COO representation.</p>
<h2>Format trade-offs</h2>
<p>As a rule of thumb, COO is only useful as an intermediate between compressed and uncompressed formats. It is fast to convert a sparse uncompressed matrix to COO format, but it is comparatively slow to convert the same matrix to CSR or CSC formats. Converting a COO matrix to a CSR or CSC matrix is fast.</p>
<p>Within SciPy, CSR matrices have been the default compression scheme for years. Arithmetic operations on CSR and CSC-compressed matrices are equally performant regardless of whether or not the operand is compressed. CSR is superior when iterating over or selecting rows, whereas CSC is superior for column operations. When in doubt, <strong>CSR is an excellent go-to compression format</strong>.</p>
<div style="display: none;">
<h1>Implementation</h1>
<p>With a solid understanding of the key compression schemes, it&#8217;s time to peruse some examples. The Python Pandas/Numpy/SciPy package family is the ecosystem of choice.</p>
<h2>Pandas</h2>
</div>
<h1>Conclusion</h1>
<p>With an understanding of data sparsity and the core matrix compression formats, you can now optimize your ML pipeline code and train models on much larger swaths of data. Let me know in the comments if you have any questions!</p>
<h1>References</h1>
<p>[1] <a href="http://www.inesc-id.pt/publications/8386/pdf">http://www.inesc-id.pt/publications/8386/pdf</a></p>
<div style="display: none;">
<p>The problems fitting the training dataset in memory will only get worse as the number of users and features grow. In the section for small scale, I alluded to switching to compressed sparse matrices. As you transition to medium scale, you will have no choice.</p>
<p>Let’s see why. Assume that you have 100K users and 10K features. To store the ratings in a dense matrix, you would need <em>100,000 users × 10,000 features × 1 byte per rating</em> = <em>1 GB.</em> That figure doesn’t seem like much, but watch what happens when the number of users and number of features both double: <em>200,000 users × 20,000 features × 1 byte per rating</em> = 4<em> GB</em>!</p>
<p>These examples show that the space complexity of matrix factorization algorithms climbs in <img loading="lazy" decoding="async" class="ql-img-inline-formula quicklatex-auto-format" title="Rendered by QuickLaTeX.com" src="http://www.datastronomy.com/wp-content/ql-cache/quicklatex.com-ea5b07fe789b780a391d065d1a775ec0_l3.svg" alt="O(|u| \cdot |f|)" width="80" height="18" />. A startup that gets to medium scale will probably be growing at 20% month-over-month, so the training dataset would consume 16 GB of RAM after four months. Sure, you can limit the number of users and features in the training set, but then you’re just buying time to avoid the inevitable.</p>
<p>Fortunately, using compressed sparse matrix representations can solve this problem. This technique takes advantage of the typical 1-2% density in rating data by compressing the ratings into a format that avoids storing missing ratings. One common compression scheme is coordinate (COO) format, where ratings are stored as a list of tuples, e.g., <em>[(user: 1, feature: 3, rating: 2), (user: 1, feature: 5, rating: 3), (user: 2, feature: 6, rating: 4), …]</em>.</p>
<p>Sparse matrices consume more memory for each rating because they lack the implicit data inherent to the location (stride), but fewer ratings overall. Let’s redo the last calculation with a sparse matrix and assume a rating density of 1%: <em>200,000 users × 20,000 features × 3 bytes per rating × 1% density</em> = a far more manageable <em>120 MB</em>.</p>
</div>

