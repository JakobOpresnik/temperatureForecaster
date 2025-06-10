export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  public: {
    Tables: {
      weather: {
        Row: {
          "Air pressure": number | null
          "Air pressure average in time interval": number | null
          Date: string
          "Humidity relative": number | null
          "Humidity relative average in time interval": number | null
          Location: string | null
          "Precipitation total in time interval": string | null
          "Solar radiation": number | null
          "Solar radiation average in time interval": number | null
          Temperature: number | null
          "Temperature average in time interval": number | null
          "Temperature dew point": number | null
          "Temperature maximum in time interval": number | null
          "Temperature minimum in time interval": number | null
          "Wind direction": number | null
          "Wind direction average in time interval": number | null
          "Wind direction maximum gust in time interval": number | null
          "Wind speed": number | null
          "Wind speed average in time interval": number | null
          "Wind speed maximum in time interval": number | null
        }
        Insert: {
          "Air pressure"?: number | null
          "Air pressure average in time interval"?: number | null
          Date: string
          "Humidity relative"?: number | null
          "Humidity relative average in time interval"?: number | null
          Location?: string | null
          "Precipitation total in time interval"?: string | null
          "Solar radiation"?: number | null
          "Solar radiation average in time interval"?: number | null
          Temperature?: number | null
          "Temperature average in time interval"?: number | null
          "Temperature dew point"?: number | null
          "Temperature maximum in time interval"?: number | null
          "Temperature minimum in time interval"?: number | null
          "Wind direction"?: number | null
          "Wind direction average in time interval"?: number | null
          "Wind direction maximum gust in time interval"?: number | null
          "Wind speed"?: number | null
          "Wind speed average in time interval"?: number | null
          "Wind speed maximum in time interval"?: number | null
        }
        Update: {
          "Air pressure"?: number | null
          "Air pressure average in time interval"?: number | null
          Date?: string
          "Humidity relative"?: number | null
          "Humidity relative average in time interval"?: number | null
          Location?: string | null
          "Precipitation total in time interval"?: string | null
          "Solar radiation"?: number | null
          "Solar radiation average in time interval"?: number | null
          Temperature?: number | null
          "Temperature average in time interval"?: number | null
          "Temperature dew point"?: number | null
          "Temperature maximum in time interval"?: number | null
          "Temperature minimum in time interval"?: number | null
          "Wind direction"?: number | null
          "Wind direction average in time interval"?: number | null
          "Wind direction maximum gust in time interval"?: number | null
          "Wind speed"?: number | null
          "Wind speed average in time interval"?: number | null
          "Wind speed maximum in time interval"?: number | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DefaultSchema = Database[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? (Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      Database[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof Database },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof Database },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends { schema: keyof Database }
  ? Database[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof Database },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof Database
  }
    ? keyof Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends { schema: keyof Database }
  ? Database[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
