import axios from "axios";
const base = "";
export async function fetchModels() {
  const response = await axios.get(`${base}/models`);
  return response.data;
}
export async function runAttack(model_id, defense, budget) {
  const response = await axios.post(`${base}/attack`, { model_id, defense, budget });
  return response.data;
}
export async function fetchTree(model_id) {
  const response = await axios.get(`${base}/tree/${model_id}`);
  return response.data;
}
