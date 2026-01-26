import GD from 'gd.js';
import { parseLevel } from 'gdparse';

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

  const extremeDemons = await gd.levels.search({ difficulty: 'Extreme Demon' }, 100);
  const cantLetGo = await gd.levels.search({ query: 'Cant Let Go' });
  const wayTooLong = await gd.levels.search({ length: 'xl' }, 100);
  const tooPopular = await gd.levels.search({ orderBy: 'downloads' }, 100);
  let bloodbath = extremeDemons[0];
  console.log(bloodbath.name); // Bloodbath
  console.log(bloodbath.stats.likes); // 1359617
  bloodbath = await bloodbath.resolve();
  console.log(bloodbath.copy.copyable); // false
  const { raw } = await bloodbath.decodeData();
  console.log(raw)
  let asdf = parseLevel(raw)
  console.log(asdf)
}

getDumbLevels();
