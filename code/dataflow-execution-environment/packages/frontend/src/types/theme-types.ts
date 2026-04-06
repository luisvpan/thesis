export type ThemeId = 'space' | 'nature' | 'robots' | 'alchemy';

export interface ThemeWorld {
  id: 'beginner' | 'intermediate' | 'final';
  name: string;
  description: string;
  icon: string;
  gradient: string;
  bgColor: string;
  textColor: string;
  levels: number;
}

export interface ThemeConfig {
  id: ThemeId;
  name: string;
  shortName: string;
  story: string;
  emoji: string;
  gradient: string;
  bgPattern: string;
  connectionStyle: {
    color: string;
    animation: string;
    description: string;
  };
  outputArea: {
    name: string;
    emoji: string;
    description: string;
  };
  worlds: ThemeWorld[];
  feedback: {
    success: string;
    error: string;
    hint: string;
  };
}
