---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

💻 I'm a computer science undergraduate, minor in statistics, at the National University of Singapore, and enrolled in the University Scholars Programme.

🧠 I'm currently researching causal reinforcement learning under Prof. Harold Soh at the Collaborative Learning and Adaptive Robots (CLeAR) lab as an undergraduate researcher.

📈 In my free time, I direct the workshops team at at NUS Statistics Society, delivering data science workshops in NUS.

💼 I previously interned as a machine learning engineer at Grab and IMDA.

{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  <p id="{{ tag }}">{{ tag }}</p>
{% endfor %}
