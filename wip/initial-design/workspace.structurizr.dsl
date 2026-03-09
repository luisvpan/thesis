workspace "The environment" "The augmented reality tangible programming environment we are making to foster computational thinking"

    model {
        k = person "Kids"
        t = person "Teacher"
        ss = softwareSystem "System" {
            cv = container "Computer Vision Manager"
            df = container "Dataflow Language Execution Environment"
            ide = container "Integrated Development Environment"
            adb = container "Activities Database" {
                tags Database
            }
        }

        k -> cv "Interacts via tangible pieces or gestures with"
        cv -> k "Shows environment to"
        
        ide -> cv "Is projected by"
        cv -> ide "Sends tangible details to"
        df -> ide "Sends execution details to"
        ide -> adb "Retrieves activities from"
        
        cv -> df "Sends language tokens to"
        # df -> cv "Responds with execution details to"
    }

    views {
        systemContext ss "system_context_view" {
            include *
            autolayout lr
        }

        container ss "whole_system_container_view" {
            include *
            autolayout lr
        }
        
        container ss "containers_only_container_view" {
            include element.type==container
            autolayout lr
        }

        styles {
            element "Element" {
                color #0773af
                stroke #0773af
                strokeWidth 7
                shape roundedbox
            }
            element "Person" {
                shape person
            }
            element "Database" {
                shape cylinder
            }
            element "Boundary" {
                strokeWidth 5
            }
            relationship "Relationship" {
                thickness 4
            }
        }
    }

}