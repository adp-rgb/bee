import { GoogleGenAI, Type } from '@google/genai';
import fs from 'fs';

const ai = new GoogleGenAI(); // Uses GEMINI_API_KEY environment variable

// 1. Read existing topic names
const rawData = JSON.parse(fs.readFileSync('topics.json', 'utf8'));

async function generateTopicDetails(topicName, category) {
  const prompt = `Generate comprehensive academic competition study facts for the topic: "${topicName}" (${category}).
Provide:
1. A clear 1-2 sentence core definition.
2. 6-10 bulleted key facts focused on high-frequency competition clues, equations, and principles.
3. 3-5 closely related topics.`;

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

async function buildFullTopicDatabase() {
  const updatedTopics = { Science: [], Geography: [] };

  for (const category of ['Science', 'Geography']) {
    const list = rawData[category] || [];
    for (const item of list) {
      const topicName = typeof item === 'string' ? item : item.name;
      console.log(`Processing [${category}]: ${topicName}...`);
      
      try {
        const details = await generateTopicDetails(topicName, category);
        updatedTopics[category].push(details);
      } catch (err) {
        console.error(`Failed to process ${topicName}:`, err);
        updatedTopics[category].push({ name: topicName, category });
      }
    }
  }

  fs.writeFileSync('topics.json', JSON.stringify(updatedTopics, null, 2));
  console.log('Successfully updated topics.json!');
}

buildFullTopicDatabase();
