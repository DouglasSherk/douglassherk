---
layout: post
title: "I Used Waterfall (And I Liked It)"
date: 2019-05-11
categories:
  - Management
source_url: https://datastronomy.com/i-used-waterfall-and-i-liked-it/
---

<p>In this article, I don&#8217;t mean to prescribe universally applicable advice, nor to downplay the importance or utility of agile, or to even suggest that more than a tiny fraction of software engineering teams should use waterfall. My purpose is simply that there&#8217;s always a right tool for the job, and that tool is not always agile.</p>



<p>In short, we&#8217;re using waterfall at Passenger AI, and it&#8217;s working. I&#8217;m going to explain our experience and attempt to rationalize it.</p>



<p>For reference, at Passenger AI, we&#8217;re building artificial intelligence software for self-driving cars to keep them clean and to protect passengers. Our offering is an edge operating system that uses computer vision, deep learning, and machine learning to track passenger behavior.</p>



<p>Now then.</p>



<div class="wp-block-image"><figure class="aligncenter"><img data-recalc-dims="1" decoding="async" src="https://i0.wp.com/www.datastronomy.com/wp-content/uploads/2019/05/image-1.png?w=750" alt="" class="wp-image-315"/></figure></div>



<p>From programming&#8217;s inception up until the 2000&#8217;s, software engineering was existentially difficult because there were no known patterns to execute on and because there was always a looming threat of total project failure. Even building a website was difficult because there were no cloud infrastructure providers or web frameworks.</p>



<p>As we rolled into the 2000&#8217;s, IaaS providers like AWS proliferated, web frameworks like Ruby on Rails launched and stabilized, and distributed systems patterns became more widely understood. Now, almost anyone can build a website or an app, the two components that are the bread and butter of most tech businesses.</p>



<p>What happened between these two eras was a shift from technological to sociological risk.</p>



<h2 class="wp-block-heading">Classes of Risk</h2>



<p>A <strong>technological risk</strong> is one where there&#8217;s uncertainty as to whether or not computers can do what&#8217;s needed, or it&#8217;s questionable that the needed technology can be built in any reasonable amount of time. Suppose that you&#8217;re training a deep neural net for an embedded system; then a technological risk is that you may need more computational resources to power that model than are available.</p>



<p>A <strong>sociological risk</strong> (&#8220;politics&#8221;) is one where communication between people, departments, and to customers can lead to a project failure. Imagine that you&#8217;re a product manager whose customers have no need for your software now, but that they will in six months; then a sociological risk is that your product is not what those customers will need in the future.</p>



<p>Technological risk dominated most software projects in the 90&#8217;s and prior, whereas sociological risk dominates most projects from the 2000&#8217;s onward.</p>



<p>That&#8217;s not to say that projects in the past didn&#8217;t involve sociological risk. Even books from the 70&#8217;s, like <em>The Mythical Man Month</em>, recognized this peril:</p>



<blockquote class="wp-block-quote is-layout-flow wp-block-quote-is-layout-flow"><p>Therefore the most important function that software builders do for their clients is the iterative extraction and refinement of the product requirements. For the truth is, the clients do not know what they want.</p><cite>Frederick P. Brooks Jr., <a href="https://en.wikipedia.org/wiki/The_Mythical_Man-Month">The Mythical Man-Month: Essays on Software Engineering</a><br><br></cite></blockquote>



<p>Let&#8217;s further explore how today&#8217;s project management confronts this danger.</p>



<h2 class="wp-block-heading">Sociological Risk</h2>



<p>Good project management mitigates risk, so modern project management focuses on resolving sociological issues like poor communication, lack of customer feedback, estimating velocity, and changing requirements. Those problems are exactly what methodologies like scrum, kanban, and XP are designed to solve. (I&#8217;ll loosely group these methodologies into &#8220;agile&#8221; from here on.)</p>



<p>Indeed, these methodologies work exceedingly well for <em>teams building apps or websites using known patterns and frameworks</em> for customers who don&#8217;t know what they want. Businesses of this type just so happen to dominate today&#8217;s tech industry.</p>



<p>The book <em>Peopleware</em> speaks in an honest way on this view:</p>



<blockquote class="wp-block-quote is-layout-flow wp-block-quote-is-layout-flow"><p>We, along with nearly everyone else involved in the high-tech endeavors, were convinced that technology was all, that whatever your problems were, there had to be a better technology solution to them. But if what you were up against was inherently sociological, better technology seemed unlikely to be much help</p><cite>Tom DeMarco &amp; Timothy Lister, <a href="https://en.wikipedia.org/wiki/Peopleware:_Productive_Projects_and_Teams">Peopleware: Productive Projects and Teams</a><br><br></cite></blockquote>



<p>While sociological risk dominates most projects today, there are still plenty in which the primary hazard is in the technology.</p>



<h2 class="wp-block-heading">Technological Risk</h2>



<p>Technological risk is a different beast because it often involves complex dependencies, front-loaded experimentation, totally unknown costs, and focused execution on known requirements.</p>



<p>Think of any deep learning startup, or any company building an operating system, or any team commercializing a research project. In none of these cases is it easy to estimate tasks, nor is there much to show customers between wireframe mocks and the final product.</p>



<p>These problems sound suspiciously similar to those that most pre-2000&#8217;s projects faced.</p>



<p>This set of technology-focused projects could operate under agile, but this methodology is not designed to meet their distinct challenges.</p>



<p>Note also that agile took off primarily because it tightened the feedback loop between engineering teams and customers. This loop enables engineering teams to function independently and to respond directly to customers rather than working through managers and product teams.</p>



<p>It goes without saying that engineers work best when given requirements and the freedom to execute on them. Then without direct contact with customers, engineers can&#8217;t make informed trade-offs on where to budget their time and effort. Managers still need to provide some requirements and direction, so it&#8217;s unclear what form those should take.</p>



<p>The question then is: what&#8217;s the best way to manage a technology-focused project if not with agile?</p>



<h2 class="wp-block-heading">Waterfall</h2>



<figure class="wp-block-image"><img data-recalc-dims="1" decoding="async" src="/assets/posts/i-used-waterfall-and-i-liked-it/gantt-chart-excel.jpg" alt="Image result for gantt chart"/></figure>



<p>If there are no metrics like revenue available to engineers, and customers aren&#8217;t providing direct feedback, then the development team needs some other signal to work with. The waterfall methodology answers this challenge by planning the important tasks and milestones ahead and then working backwards to establish the timeline required for success. A popular way of visualizing these tasks and milestones is using a Gantt chart (above).</p>



<p>This process involves talking with the engineers who will be implementing the project to make sure that they understand the scope, the goals, and to get reasonable estimates. This procedure also requires an architect, or small team of architects, to create and maintain a consistent project architecture.</p>



<p>The advantage of this approach is that engineers then know how much time and effort they should allocate for each task. The per-task time budget signals that task&#8217;s relative importance. The point isn&#8217;t to impose arbitrary deadlines, to be inflexible, or to shame engineers for under- or overshooting, but to <em>know at a glance if the project is on-time or if a task is blocked</em> and to shift resources or to descope accordingly.</p>



<p>It&#8217;s fashionable to resent waterfall project management, and with good reason. It should be no secret from reading this article&#8217;s title that I surreptitiously see otherwise.</p>



<p>This pattern is exactly what we&#8217;re using at Passenger AI, and it&#8217;s working incredibly well. We even use a Gantt chart as described above. We tried agile in various manifestations but found that it had too much ceremony and it didn&#8217;t answer at a glance the important questions we asked of it.</p>



<p>I have my concerns, and the system isn&#8217;t perfect, but our team is getting as much done as anyone could possibly ask of them, and everyone is happy with our choice.</p>

