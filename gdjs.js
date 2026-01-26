import GD from 'gd.js';
import { parseLevel } from 'gdparse';
import fs from 'fs';

const gd = new GD();


const getDumbLevels = async () => {
  // let level = await gd.levels.search({ query: '131860273' });
  // console.log(level);
  // console.log(level.constructor.name);
  // level = await level.resolve();
  // console.log(level.copy.copyable);
  // const { raw: rawData } = await level.decodeData();
  // const parsedData = parseLevel(rawData);
  // console.log(parsedData);

  const levels = await gd.levels.search({ difficulty: 'Easy' }, 100);
  console.log(levels.length)
  for (let i = 0; i < levels.length; i++) {
    let level = levels[i];
    try {
      level = await level.resolve();
      const { raw } = await level.decodeData();
      let parsedLevel = parseLevel(raw)
      const stream = fs.createWriteStream(`./levels/${level.id}.json`);
      stream.write('{\n');
      stream.write(`\t"id": ${level.id},\n`);
      stream.write(`\t"Creator": {"id": ${level.creator.id}},\n`);
      stream.write(`\t"description": "${level.description}",\n`);
      stream.write(`\t"diamonds": ${level.diamonds},\n`)
      stream.write(`\t"Difficulty": {"level": "${level.difficulty.level.pretty}", "requestedStars": ${level.difficulty.requestedStars}, "stars": ${level.difficulty.stars}},\n`);
      stream.write(`\t"stats": {"downloads": ${level.stats.downloads}, "length": {"pretty": "${level.stats.length.pretty}", "raw": "${level.stats.length.raw}"}, "likes": ${level.stats.likes}, "objects": ${level.stats.objects}}`);
      stream.write('}');
      stream.end();
    } catch (e) {
      console.log(`Fetching level data failed for level ${level.id}`);
    }
  }
}

getDumbLevels();
