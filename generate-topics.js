import { GoogleGenAI, Type } from '@google/genai';
import fs from 'fs';
import axios from 'axios';
import * as cheerio from 'cheerio';

const ai = new GoogleGenAI(); // Uses GEMINI_API_KEY environment variable

// 1. Resource Mapping for IAC / IGB Competition Domains
const SOURCE_URLS = {
  Science: [
    'https://iacompetitionsasia.com/resources/',
    'https://www.iacompetitions.com/ems-national-science-bee-past-questions/'
  ],
  Geography: [
    'https://www.internationalgeographybee.com/asia/resources/',
    'https://iacompetitionsasia.com/resources/'
  ]
};

// 2. Fetch and strip text content from target URLs
async function fetchPageText(url) {
  try {
    const { data } = await axios.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
      timeout: 10000
    });
    const $ = cheerio.load(data);
    $('script, style, nav, footer, header').remove();
    return $('body').text().replace(/\s+/g, ' ').trim().slice(0, 8000);
  } catch (err) {
    console.warn(`Could not scrape ${url}: ${err.message}`);
    return '';
  }
}

// 3. Pre-load combined site context per subject
async function loadTargetContexts() {
  const contexts = { Science: '', Geography: '' };
  
  for (const domain of SOURCE_URLS.Science) {
    console.log(`Scraping Science context from ${domain}...`);
    contexts.Science += (await fetchPageText(domain)) + '\n';
  }
  
  for (const domain of SOURCE_URLS.Geography) {
    console.log(`Scraping Geography context from ${domain}...`);
    contexts.Geography += (await fetchPageText(domain)) + '\n';
  }
  
  return contexts;
}

// 4. Generate structured JSON via Gemini
async function generateTopicDetails(topicName, category, sourceContext) {
  const prompt = `You are a Quizbowl/IAC Academic Competition preparation engine.
Extract and format competition-grade study material for the topic: "${topicName}" within category "${category}".

Reference Text from Official Resources:
"""
${sourceContext}
"""

Output Format Instructions:
1. Provide a concise 1-2 sentence definition.
2. Provide 6-10 high-frequency bullet points (clues, formulas, laws, or historical facts tested in IAC/IGB bees).
3. Provide 3-5 closely related topics.`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
    config: {
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          definition: { type: Type.STRING },
          key_facts: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
          },
          related_topics: {
            type: Type.ARRAY,
            items: { type: Type.STRING }
          }
        },
        required: ['name', 'definition', 'key_facts', 'related_topics']
      }
    }
  });

  return JSON.parse(response.text);
}

// 5. Main Execution Loop
async function processAllTopics() {
  const rawData = JSON.parse(fs.readFileSync('topics.json', 'utf8'));
  const contexts = await loadTargetContexts();
  const updatedTopics = { Science: [], Geography: [] };

  for (const category of ['Science', 'Geography']) {
    const topicList = rawData[category] || [];
    const domainContext = contexts[category];

    for (const item of topicList) {
      const topicName = typeof item === 'string' ? item : item.name;
      console.log(`Processing [${category}]: ${topicName}...`);

      try {
        const parsedData = await generateTopicDetails(topicName, category, domainContext);
        updatedTopics[category].push(parsedData);
      } catch (err) {
        console.error(`Error processing ${topicName}:`, err.message);
        updatedTopics[category].push(typeof item === 'object' ? item : { name: topicName });
      }
    }
  }

  fs.writeFileSync('topics.json', JSON.stringify(updatedTopics, null, 2));
  console.log('Successfully updated topics.json with IAC & IGB resource data!');
}

processAllTopics();
