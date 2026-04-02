---
layout: default
title: Home
---

<div class="home-intro">
  <p class="eyebrow">Software Engineering Blog</p>
  <h1>Doug Sherk</h1>
  <p class="lead">
    Writing about software engineering, machine learning, tooling, and building reliable systems.
  </p>
</div>

## Posts

{% if site.posts.size > 0 %}
  <ul class="post-list">
    {% for post in site.posts %}
      <li class="post-item">
        <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
        {% if post.excerpt %}
          <p class="post-excerpt">{{ post.excerpt | strip_html | truncate: 140 }}</p>
        {% endif %}
      </li>
    {% endfor %}
  </ul>
{% else %}
  <p>No posts yet.</p>
{% endif %}
