import type { FoodType } from '@/types/card-types';

export const FOOD_EMOJI: Record<FoodType, string> = {
  manzana: '🍎',
  hamburguesa: '🍔',
  uvas: '🍇',
  pasta: '🍝',
  pera: '🍐',
  naranja: '🍊',
};

export function foodEmoji(food: FoodType): string {
  return FOOD_EMOJI[food] ?? '🍽️';
}
