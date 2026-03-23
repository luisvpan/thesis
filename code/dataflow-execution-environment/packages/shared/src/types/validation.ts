export type ValidationError = {
  code: string;
  message: string;
  childMessage?: string;
  nodeId?: string;
  suggestion?: string;
  example?: string;
};

export type ValidationResult = {
  success: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
};

export type MissingInput = {
  port: number;
  description: string;
  childMessage: string;
};
